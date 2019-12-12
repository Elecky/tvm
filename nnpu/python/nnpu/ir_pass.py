'''
additional ir pass for nnpu, to transform ir before lowering
'''

from .environment import get_env
import tvm
from .helper import dtype_bytes as get_dtype_bytes, get_access_ptr
from topi import util
from .utils import *
from .intrins import make_intrin_call

tvm_zero = tvm.const(0, 'uint32')

def _match_pragma(stmt, key):
    """Internal helper to match stmt to pragma stmt.

    Parameters
    ----------
    stmt : Stmt
        The AttrStmt

    key : str
        The pragma key
    """
    return ((stmt.attr_key == "pragma_" + key) or
            (stmt.attr_key == "pragma_scope" and stmt.value.value == key))

def to_const_ints(exprs):
    ret = list()
    for expr in exprs:
        ret.append(util.get_const_int(expr))
    return ret

# some helper functions
def mark_coproc_scope(stmt, pid, is_uop=False):
    irb = tvm.ir_builder.create()
    env = get_env()
    irb.scope_attr(env.nnpu_axis, "nnpu_function", 0)
    irb.scope_attr(get_env().nnpu_axis, "coproc_scope", pid)
    if (is_uop):
        irb.scope_attr(env.nnpu_axis, "coproc_uop_scope", pid)
    irb.emit(stmt)
    body = irb.get()
    return body

def get_mode_code(dtype, dtype_out = None):
    mode2code = {'n': 0, 'inc': 1, 'dec': 2, 'w': 3}
    env = get_env()
    assert dtype in [env.cfg['dtype_n'], env.cfg['dtype_w']], 'invalid dtype'

    dtype_out = dtype if dtype_out is None else dtype_out
    if (dtype == env.cfg['dtype_n']):
        mode = 'n' if dtype_out == env.cfg['dtype_w'] else 'inc'
    else:
        mode = 'w' if dtype_out == env.cfg['dtype_w'] else 'dec'

    return mode2code[mode]

def _fold(src_shape, src_strides, dst_shape, dst_strides, pad_before = None, pad_after = None):
    #if (pad_after or pad_before):
    #    raise NotImplementedError('padding is not supported right now')
    
    ndim = len(src_shape)
    assert len(src_shape) == len(dst_shape), \
        'ndim of copying source and destination not matching, {0} vs {1}'.format(
            len(src_shape), len(dst_shape)
        )
    
    for i in range(len(src_shape)):
        if (pad_after and pad_after):
            assert util.equal_const_int(
                dst_shape[i] - src_shape[i] - pad_before[i] - pad_after[i], 0), \
                'shape of copying source and destination not matching even with padding'
        else:
            assert util.equal_const_int(dst_shape[i] - src_shape[i], 0), \
                'shape of copying source and destination not matching'
    # now fold dimensions
    s_shape = []
    s_strides = []
    d_shape = []
    d_strides = []
    p_before = []
    p_after = []
    t_size = 1
    ts_stride = src_strides[ndim-1]
    td_stride = dst_strides[ndim-1]  # both strides are actually 1
    
    index = ndim - 1
    while (index >= 0):
        # stop folding when any padding needed from here
        #if (not util.equal_const_int(src_shape[index] - dst_shape[index], 0)):
        #    break
        
        if (index > 0 and
            util.equal_const_int(dst_shape[index] - src_shape[index], 0) 
            and
            util.equal_const_int(dst_shape[index - 1] - src_shape[index - 1], 0)
            and
            util.equal_const_int(
                src_shape[index] * src_strides[index] - src_strides[index - 1], 0)
            and 
            util.equal_const_int(
                dst_shape[index] * dst_strides[index] - dst_strides[index - 1], 0)
            ):
            # the conditions check:
            # index is not the highest dimension,
            # current dimension will not be padded,
            # next higher dimension will not be padded,
            # current dimension is continous with next higher dimension in both 
            # source and destination and could be folded.

            t_size = t_size * src_shape[index]
        else:
            # append current group
            s_shape.append(t_size * src_shape[index])
            d_shape.append(t_size * dst_shape[index])
            s_strides.append(ts_stride)
            d_strides.append(td_stride)
            if (pad_before):
                p_before.append(pad_before[index])
            if (pad_after):
                p_after.append(pad_after[index])
            
            # next group initial value
            if (index > 0):
                t_size = 1
                ts_stride = src_strides[index - 1]
                td_stride = dst_strides[index - 1]
        index = index - 1
    # reverse all arrays
    s_shape.reverse()
    s_strides.reverse()
    d_shape.reverse()
    d_strides.reverse()
    p_before.reverse()
    p_after.reverse()
    p_before = None if (len(p_before) == 0) else p_before
    p_after = None if len(p_after) == 0 else p_after

    return s_shape, s_strides, d_shape, d_strides, p_before, p_after

def _build_copy(dst, src,
                src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after,
                create_copy_ir, create_memset, memcpy_dim=1):
    '''
    create_copy_ir: a function that creates memory copy ir, has signature:
        void(dst_idx /*index of destination*/,
             dst_stride /*stride of destination*/,
             src_idx /*offset of source*/,
             src_stride /*stride of source*/,
             nUnit /*how many elements to copy*/)
             
    create_memset: a function that creates memset ir, has signature:
        void(index /*index of destination that memset should begin at*/,
             nUnit /*number of elements to set*/,
             stride /*stride between two consecutive elements*/)
    '''
    ndim = len(src_shape)

    def build_padding(dst_base, level, shape, strides):
        if (level == len(shape) - 1):
            body = create_memset(dst_base, shape[-1], dst_strides[-1])
            irb.emit(body)
        else:
            var = tvm.var('i.pad.{0}'.format(level))
            body = build_padding(dst_base + var * strides[level], level + 1, shape, strides)
            loop = tvm.make.For(var, 0, shape[level], 0, 0, body)  # fortype = serial)
            return loop

    def _build(src_base, dst_base, level):
        '''
            src_base: index of source (in element count)
            dst_base: index of destination (in element count)
            level: the recursive level, also indicates the axis.
        '''
        if (level >= ndim - memcpy_dim):
            if (level == ndim - 1):
                # if it's the last dimension, use scalar arguments
                body = create_copy_ir(
                        dst_base,
                        dst_strides[-1],
                        src_base,
                        src_strides[-1],
                        src_shape[-1])
            else:
                body = create_copy_ir(
                        dst_base,
                        dst_strides[-memcpy_dim:],
                        src_base + (0 if not pad_before else pad_before[level]) * dst_strides[level],
                        src_strides[-memcpy_dim:],
                        src_shape[-memcpy_dim:])
            body = tvm.make.Evaluate(body)
            return body
        else:
            irb = tvm.ir_builder.create()

            if (pad_before and pad_before[level] != 0):
                irb.emit(build_padding(dst_base, 0, [pad_before[level]] + dst_shape[level + 1:], dst_strides[level:]))
                
            # iterate from 0 to src_shape[level]
            var = tvm.var('i{0}'.format(level))
            body = _build(src_base + var * src_strides[level],
                          dst_base + (var + (0 if not pad_before else pad_before[level])) \
                                     * dst_strides[level],
                          level + 1)
            loop = tvm.make.For(var, 0, src_shape[level], 0, 0, body)  # fortype = serial)
            irb.emit(loop)

            if (pad_after and pad_after[level] != 0):
                irb.emit(build_padding(dst_base + (src_shape[level] + pad_before[level]) * dst_strides[level],
                                       0, [pad_after[level]] + dst_shape[level + 1:], dst_strides[level:]))

            return irb.get()
    return _build(0, 0, 0)

def inject_dma_intrin(stmt_in):
    env = get_env()

    def _error(*args):
        raise NotImplementedError('DMA copy dont support padding')

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        #print('inject_copy called')
        if (pad_after or pad_before):
            raise NotImplementedError('padding is not supported right now')

        assert src.dtype == dst.dtype, 'dtype of copying source and destination does not match, \
            {0} vs {1}'.format(src.dtype, dst.dtype)
        
        dtype_bytes = get_dtype_bytes(src.dtype)
        
        ndim = len(src.shape)
        
        assert util.equal_const_int(src.strides[ndim - 1], 1), \
            'stride of last dimension must be 1'
        assert util.equal_const_int(dst.strides[ndim - 1], 1), \
            'stride of last dimension must be 1'

        if (src.scope == 'global' and dst.scope == env.dram_scope):
            src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)

            body = _build_copy(
                        dst, src,
                        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after,
                        # lambda to create DMALoad IR:
                        lambda dst_idx, dst_stride, src_idx, src_stride, nUnit:
                            tvm.call_llvm_intrin_with_side_effect(
                                'void', "llvm.NNPU.DMALoad", tvm_zero,
                                src.data, (src_idx + src.elem_offset) * dtype_bytes,
                                get_access_ptr(dst, env, 'w') + dst_idx * dtype_bytes,
                                nUnit * dtype_bytes),
                        _error
                        )
            body = mark_coproc_scope(body, env.get_pid(env.pid_dma_copy))
            return body
        elif (src.scope == env.dram_scope and dst.scope == 'global'):
            assert not (pad_after or pad_before), \
                'padding is not supported when copying to global'
            src_shape, src_strides, dst_shape, dst_strides, _, _ = \
                _fold(src.shape, src.strides, dst.shape, dst.strides)
            
            body = _build_copy(
                        dst, src,
                        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after,
                        lambda dst_idx, dst_stride, src_idx, src_stride, nUnit:
                            tvm.call_llvm_intrin_with_side_effect(
                                'void', "llvm.NNPU.DMAStore", tvm_zero,
                                dst.data, (dst_idx + dst.elem_offset) * dtype_bytes,
                                get_access_ptr(src, env, 'r') + src_idx * dtype_bytes,
                                nUnit * dtype_bytes),
                        _error
                        )
            body = mark_coproc_scope(body, env.get_pid(env.pid_dma_copy))
            return body
        else:
            raise ValueError('donnot support copy from {0} to {1}'.format(
                src.scope, dst.scope
            ))
        pass

    return tvm.ir_pass.InjectCopyIntrin(stmt_in, env.dma_copy_pragma, _inject_copy)

def inject_dmacopy2buf_intrin(stmt_in):
    env = get_env()

    def _error(*args):
        raise NotImplementedError('DMA copy dont support padding')

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        assert src.dtype == dst.dtype, 'dtype of copying source and destination does not match, \
            {0} vs {1}'.format(src.dtype, dst.dtype)
        
        dtype_bytes = get_dtype_bytes(src.dtype)
        
        ndim = len(src.shape)
        
        assert util.equal_const_int(src.strides[ndim - 1], 1) and \
                util.equal_const_int(dst.strides[ndim - 1], 1), \
            'stride of last dimension must be 1, ie, data must be compact'

        if (src.scope == 'global' and env.is_scratchpad_scope(dst.scope)):
            src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)

            def create_ir(dst_idx, dst_strides, src_idx, src_strides, extends):
                if (not isinstance(extends, list)):
                    return tvm.call_intrin(
                                'int32', "NNPU.DMABufLoad",
                                src.data, (src_idx + src.elem_offset) * dtype_bytes, 0,
                                get_access_ptr(dst, env, 'w') + dst_idx * dtype_bytes, 0,
                                extends * dtype_bytes, 1)
                else:
                    assert len(extends) == 2, 'only 1 or 2 dimension DMA copy is supported'
                    return tvm.call_intrin(
                                'int32', "NNPU.DMABufLoad",
                                src.data, (src_idx + src.elem_offset) * dtype_bytes, src_strides[0] * dtype_bytes,
                                get_access_ptr(dst, env, 'w') + dst_idx * dtype_bytes, dst_strides[0] * dtype_bytes,
                                extends[1] * dtype_bytes, extends[0])
            body = _build_copy(
                        dst, src,
                        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after,
                        # lambda to create DMALoad IR:
                        create_ir, _error, 2)
            body = mark_coproc_scope(body, env.get_pid(env.pid_dma_copy), True)
            return body
        elif (env.is_scratchpad_scope(src.scope) and dst.scope == 'global'):
            assert not (pad_after or pad_before), \
                'padding is not supported when copying to global'
            src_shape, src_strides, dst_shape, dst_strides, _, _ = \
                _fold(src.shape, src.strides, dst.shape, dst.strides)
            
            def create_ir(dst_idx, dst_strides, src_idx, src_strides, extends):
                if (not isinstance(extends, list)):
                    return tvm.call_intrin(
                                'int32', "NNPU.DMABufStore",
                                dst.data, (dst_idx + dst.elem_offset) * dtype_bytes, 0,
                                get_access_ptr(src, env, 'r') + src_idx * dtype_bytes, 0,
                                extends * dtype_bytes, 1)
                else:
                    assert len(extends) == 2, 'only 1 or 2 dimension DMA copy is supported'
                    return tvm.call_intrin(
                                'int32', "NNPU.DMABufStore",
                                dst.data, (dst_idx + dst.elem_offset) * dtype_bytes, dst_strides[0] * dtype_bytes,
                                get_access_ptr(src, env, 'w') + src_idx * dtype_bytes, src_strides[0] * dtype_bytes,
                                extends[1] * dtype_bytes, extends[0])

            body = _build_copy(
                        dst, src,
                        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after,
                        create_ir, _error, 2)
            body = mark_coproc_scope(body, env.get_pid(env.pid_dma_copy), True)
            return body
        else:
            raise ValueError('donnot support copy from {0} to {1}'.format(
                src.scope, dst.scope
            ))
        pass

    return tvm.ir_pass.InjectCopyIntrin(stmt_in, env.dma_copy_to_buf, _inject_copy)

def inject_scratchpad_ls(stmt_in):
    env = get_env()

    def _error(*args):
        raise NotImplementedError('Scratchpad Load/Store dont support padding')

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        if (pad_after or pad_before):
            raise NotImplementedError('padding is not supported right now')
        
        if ((pad_before and not util.equal_const_int(pad_before[-1], 0)) or 
            (pad_after and not util.equal_const_int(pad_after[-1], 0))):
            raise ValueError('can not pad last dimension')

        assert src.dtype == dst.dtype, 'dtype of copying source and destination does not match, \
            {0} vs {1}'.format(src.dtype, dst.dtype)
        
        dtype_bytes = get_dtype_bytes(src.dtype)

        ndim = len(src.shape)
        assert util.equal_const_int(src.strides[ndim - 1], 1), \
            'stride of last dimension must be 1'
        assert util.equal_const_int(dst.strides[ndim - 1], 1), \
            'stride of last dimension must be 1'

        if (src.scope == env.dram_scope and env.is_scratchpad_scope(dst.scope)):
            src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)
            
            body = _build_copy(
                        dst, src,
                        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after,
                        lambda dst_idx, dst_stride, src_idx, src_stride, nUnit:
                        # TODO: here should assert that both src_stride & dst_stride equals 1!
                            tvm.call_llvm_intrin_with_side_effect(
                                'void', "llvm.NNPU.ScratchpadLoad", tvm_zero,
                                get_access_ptr(src, env, 'r') + src_idx * dtype_bytes,
                                get_access_ptr(dst, env, 'w') + dst_idx * dtype_bytes,
                                nUnit * dtype_bytes),
                        _error
                    )
            body = mark_coproc_scope(body, env.get_pid(env.pid_dma_copy))
            return body

        elif (env.is_scratchpad_scope(src.scope) and dst.scope == env.dram_scope):
            #assert not (pad_after or pad_before), \
            #    'padding is not supported when copying to global'
            src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)
            body = _build_copy(
                        dst, src,
                        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after,
                        lambda dst_idx, dst_stride, src_idx, src_stride, nUnit:
                            tvm.call_llvm_intrin_with_side_effect(
                                'void', "llvm.NNPU.ScratchpadStore", tvm_zero,
                                get_access_ptr(dst, env, 'w') + dst_idx * dtype_bytes,
                                get_access_ptr(src, env, 'r') + src_idx * dtype_bytes,
                                nUnit * dtype_bytes),
                        _error
                    )
            body = mark_coproc_scope(body, env.get_pid(env.pid_dma_copy))
            return body
        else:
            raise ValueError('donnot support copy from {0} to {1}'.format(
                src.scope, dst.scope
            ))
        pass

    return tvm.ir_pass.InjectCopyIntrin(stmt_in, env.scratchpad_ls, _inject_copy)
    #src_shape = [2, 2, 16]

def inject_scratchpad_copy(stmt_in):
    env = get_env()

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        #print('inject_scratchpad_copy called')
        if (pad_value):
            print('using zero padding now!!')
            pad_value = tvm.const(0, 'float64')

        if ((pad_before and not util.equal_const_int(pad_before[-1], 0)) or 
            (pad_after and not util.equal_const_int(pad_after[-1], 0))):
            raise ValueError('can not pad last dimension')
        
        assert src.dtype == dst.dtype, 'dtype of copying source and destination does not match, \
            {0} vs {1}'.format(src.dtype, dst.dtype)
        
        # check memory scope
        assert env.is_scratchpad_scope(src.scope), 'source buffer scope is not scratchpad'
        assert env.is_scratchpad_scope(dst.scope), 'destination buffer scope is not scratchpad'

        dtype_bytes = get_dtype_bytes(src.dtype)
        
        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)
        compact = util.equal_const_int(src.strides[-1], 1) and util.equal_const_int(dst.strides[-1], 1)        
        def create_ir(dst_idx, dst_strides, src_idx, src_strides, extends):
            if (not isinstance(extends, list)):
                if (util.equal_const_int(src_strides, 1) and util.equal_const_int(dst_strides, 1)):
                    # if it's actually compact
                    return tvm.call_intrin(
                            'int32', "NNPU.ScratchpadCopy",
                            get_access_ptr(dst, env, 'w') + dst_idx * dtype_bytes, 0,
                            get_access_ptr(src, env, 'r') + src_idx * dtype_bytes, 0,
                            dtype_bytes * extends, 1)
                else:
                    return tvm.call_intrin(
                                'int32', "NNPU.ScratchpadCopy",
                                get_access_ptr(dst, env, 'w') + dst_idx * dtype_bytes, dst_strides * dtype_bytes,
                                get_access_ptr(src, env, 'r') + src_idx * dtype_bytes, src_strides * dtype_bytes,
                                dtype_bytes, extends)
            else:
                assert len(extends) == 2, 'only 1 or 2 dimension DMA copy is supported'
                assert compact, 'only when last dimension is compact can do 2D copy'
                return tvm.call_intrin(
                            'int32', "NNPU.ScratchpadCopy",
                            get_access_ptr(dst, env, 'w') + dst_idx * dtype_bytes, dst_strides[0] * dtype_bytes,
                            get_access_ptr(src, env, 'w') + src_idx * dtype_bytes, src_strides[0] * dtype_bytes,
                            extends[1] * dtype_bytes, extends[0])
        body = _build_copy(
                    dst, src,
                    src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after,
                    create_ir,
                    lambda index, nUnit, stride:
                        tvm.call_llvm_intrin_with_side_effect(
                            "void", 'llvm.NNPU.Memset', tvm_zero,
                            get_access_ptr(dst, env, 'w') + index * dtype_bytes, 
                            nUnit, 
                            stride * dtype_bytes,
                            pad_value, 
                            get_mode_code(dst.dtype)
                        ),
                    2 if compact else 1)

        body = mark_coproc_scope(body, env.get_pid(env.pid_scratchpad_copy(dst.scope)), True)
        return body
    
    return tvm.ir_pass.InjectCopyIntrin(stmt_in, env.scratchpad_copy, _inject_copy)

def inject_accTobuffer(stmt_in):
    env = get_env()

    def _error(*args):
        raise NotImplementedError('AccToBuffer copy dont support padding yet.')

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        #print('inject_scratchpad_copy called')

        if ((pad_before and not util.equal_const_int(pad_before[-1], 0)) or 
            (pad_after and not util.equal_const_int(pad_after[-1], 0))):
            raise ValueError('can not pad last dimension')
        
        assert util.equal_const_int(src.strides[-1], 1) \
               and util.equal_const_int(dst.strides[-1], 1), \
                'when copying from acc-buffer to scratchpad, last dimension must be compact'
        
        assert src.dtype == dst.dtype or src.dtype == env.cfg['dtype_w'] and dst.dtype == env.cfg['dtype_n'], \
            'copy from acc-buffer to scratchpad can only keep dtype or decrease from dtype_w to dtype_n, \
given = {0} vs {1}'.format(src.dtype, dst.dtype)
        
        # check memory scope
        assert src.scope == env.acc_scope, 'source scope can only be accumulation buffer'
        assert env.is_scratchpad_scope(dst.scope), 'dst buffer scope is not scratchpad'

        dtype_bytes = get_dtype_bytes(src.dtype)
        
        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)
        body = _build_copy(
                    dst, src,
                    src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after,
                    lambda dst_idx, dst_stride, src_idx, src_stride, nUnit:
                        tvm.call_intrin(
                            'int32', "NNPU.CopyAccToBuffer",
                            get_access_ptr(dst, env, 'w') + dst_idx * dtype_bytes,
                            get_access_ptr(src, env, 'r') + src_idx * dtype_bytes,
                            nUnit,
                            get_mode_code(src.dtype, dst.dtype)),
                    _error
                )
        body = mark_coproc_scope(body, env.get_pid(env.pid_acc2buf_copy), True)
        return body
    
    return tvm.ir_pass.InjectCopyIntrin(stmt_in, env.copy_acc2buf, _inject_copy)

def cpu_access_rewrite(stmt_in):
    """Detect CPU access to VTA buffer and get address correctly. copied from VTA.

    VTA's buffer is an opaque handle that do not
    correspond to address in CPU.
    This pass detect CPU access and rewrite to use pointer
    returned VTABufferCPUPtr for CPU access.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    env = get_env()
    rw_info = {}
    def _post_order(op):
        if isinstance(op, tvm.stmt.Allocate):
            buffer_var = op.buffer_var
            if not buffer_var in rw_info:
                return None
            new_var = rw_info[buffer_var]
            let_stmt = tvm.make.LetStmt(
                new_var, tvm.call_extern(
                    "handle", "NNPUBufferCPUPtr",
                    # env.dev.command_handle,
                    buffer_var), op.body)
            alloc = tvm.make.Allocate(
                buffer_var, op.dtype, op.extents,
                op.condition, let_stmt)
            del rw_info[buffer_var]
            return alloc
        elif isinstance(op, tvm.expr.Load):
            buffer_var = op.buffer_var
            if not buffer_var in rw_info:
                rw_info[buffer_var] = tvm.var(
                    buffer_var.name + "_ptr", "handle")
            new_var = rw_info[buffer_var]
            return tvm.make.Load(op.dtype, new_var, op.index)
        elif isinstance(op, tvm.stmt.Store):
            buffer_var = op.buffer_var
            if not buffer_var in rw_info:
                rw_info[buffer_var] = tvm.var(
                    buffer_var.name + "_ptr", "handle")
            new_var = rw_info[buffer_var]
            return tvm.make.Store(new_var, op.value, op.index)
        else:
            raise RuntimeError("not reached")
    stmt = tvm.ir_pass.IRTransform(
        stmt_in, None, _post_order, ["Allocate", "Load", "Store"])
    for buffer_var, new_var in rw_info.items():
        stmt = tvm.make.LetStmt(
            new_var, tvm.call_extern(
                    "handle", "NNPUBufferCPUPtr",
                    # env.dev.command_handle,
                    buffer_var), stmt)
    return stmt


def lift_alloc_to_scope_begin(stmt_in):
    """Lift allocate to beginning of the current scope. copied from VTA.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    lift_stmt = [[]]
    def _merge_block(slist, body):
        for op in slist:
            if op.body == body:
                body = op
            elif isinstance(op, tvm.stmt.Allocate):
                body = tvm.make.Allocate(
                    op.buffer_var, op.dtype,
                    op.extents, op.condition, body)
            elif isinstance(op, tvm.stmt.AttrStmt):
                body = tvm.make.AttrStmt(
                    op.node, op.attr_key, op.value, body)
            elif isinstance(op, tvm.stmt.For):
                body = tvm.make.For(
                    op.loop_var, op.min, op.extent, op.for_type,
                    op.device_api, body)
            else:
                raise RuntimeError("unexpected op")
        del slist[:]
        return body

    def _pre_order(op):
        if isinstance(op, tvm.stmt.For):
            lift_stmt.append([])
        elif isinstance(op, tvm.stmt.AttrStmt):
            if op.attr_key == "virtual_thread":
                lift_stmt.append([])

        return None

    def _post_order(op):
        if isinstance(op, tvm.stmt.Allocate):
            lift_stmt[-1].append(op)
            return op.body
        elif isinstance(op, tvm.stmt.AttrStmt):
            if op.attr_key == "storage_scope":
                lift_stmt[-1].append(op)
                return op.body
            elif op.attr_key == "virtual_thread":
                return _merge_block(lift_stmt.pop() + [op], op.body)
            return op
        elif isinstance(op, tvm.stmt.For):
            return _merge_block(lift_stmt.pop() + [op], op.body)
        else:
            raise RuntimeError("not reached")
    stmt = tvm.ir_pass.IRTransform(
        stmt_in, _pre_order, _post_order, ["Allocate", "AttrStmt", "For"])
    assert len(lift_stmt) == 1
    return _merge_block(lift_stmt[0], stmt)

def load_cast_rewrite(stmt_in):
    """detect load and cast in NNPU device scope IR, and replace it as corresponding LLVM Intrinsic.

    there can be some load and cast operation even after tensorize. 
    in ROI Pooling, for example, has some Cast(Load(...)) nodes to compute the ROI regions.
    this pass detects those node and convert to corresponding LLVM Intrinsic function.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    env = get_env()
    def post_order_(op):
        if (isinstance(op, tvm.expr.Cast) and isinstance(op.value, tvm.expr.Load)):
            load = op.value
            # create tvm_access_ptr call.
            e_dtype = tvm.call_pure_intrin(load.dtype, 'type_annotation')
            data = load.buffer_var
            ptr = tvm.call_intrin('int32', 'tvm_access_ptr', e_dtype, data, load.index, 1, 1)
            # create llvm intrin call to do load and cast.
            mode = get_mode_code(load.dtype)
            value = make_intrin_call('int32', 'MoveFromBuf', ptr, mode)
            return value

        elif (isinstance(op, tvm.expr.Load)):
            # NOTE: currently, we always expect the Load is inside Cast, so do nothing here.
            pass
        return None
    
    def find_nnpu_device_ir(op):
        '''find NNPU device ir, and use post_order_ to mutate it.
        '''
        if (isinstance(op, tvm.stmt.AttrStmt) and
            op.attr_key == 'nnpu_function'):
            body = tvm.ir_pass.IRTransform(
                    op.body, None, post_order_, ['Cast', 'Load'])
            return tvm.make.AttrStmt(
                    op.node, op.attr_key, op.value, body)
        return None

    return tvm.ir_pass.IRTransform(
            stmt_in, find_nnpu_device_ir, None, ["AttrStmt"])

'''
passes to do custom tensorize
'''
mode2code = {'n': 0, 'inc': 1, 'dec': 2, 'w': 3}
def get_mode_code(dtype_in, dtype_out):
    env = get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    if (dtype_in == dtype_n):
        if (dtype_out == dtype_n):
            return mode2code['n']
        elif (dtype_out == dtype_w):
            return mode2code['inc']
    elif (dtype_in == dtype_w):
        if (dtype_out == dtype_n):
            return mode2code['dec']
        elif (dtype_out == dtype_w):
            return mode2code['w']
    raise ValueError('invalid dtype')

def create_access_ptr(dtype, buffer_var, index, extent, access_mask):
    e_dtype = tvm.call_pure_intrin(dtype, 'type_annotation')
    data = buffer_var
    mask = 0
    for value in access_mask:
        if value == "r":
            mask = mask | 1
        elif value == "w":
            mask = mask | 2
        else:
            raise ValueError('invalid mask')
    ptr = tvm.call_intrin('int32', 'tvm_access_ptr', e_dtype, data, index, extent, mask)
    return ptr

def annotate_coproc_scope(stmt_in):
    """Pass to insert ALU instruction.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    env = get_env()
    def _do_fold(stmt):
        if _match_pragma(stmt, "nnpu.vector"):
            return mark_coproc_scope(stmt, env.get_pid(env.pid_vector_compute), True)
        elif _match_pragma(stmt, 'nnpu.im2col'):
            return mark_coproc_scope(stmt, env.get_pid(env.pid_scratchpad_copy(env.get_scope('buffer0'))), True)
        return stmt

    stmt_out = tvm.ir_pass.IRTransform(
        stmt_in, None, _do_fold, ["AttrStmt"])

    return stmt_out

def custom_tensorize(stmt_in):
    import ast

    def binary_(stmt, params):
        # assert stmt.for_type == 1, 'loop must be parallel'
        # stmt = tvm.ir_pass.CanonicalSimplify(stmt)
        # stmt = tvm.ir_pass.Simplify(stmt)
        # print (stmt)
        assert isinstance(stmt, tvm.stmt.For), 'invalid AST, not For loop'
        loop_var = stmt.loop_var
        indices = [loop_var]
        assert util.equal_const_int(stmt.min, 0), 'invalid loop start'
        ext = util.get_const_int(stmt.extent)
        if ext != params['size']:
            raise RuntimeError('invalid AST, loop with extent {0} is exptected, got {1}'.format(params['size'], ext))
        body = stmt.body
        if (not isinstance(body, tvm.stmt.Store)):
            raise RuntimeError('invalid AST, Store expected')
        dst_var = body.buffer_var
        dst_idx = body.index
        assert util.equal_const_int(body.predicate, 1) or not body.predicate.defined(), 'predicate is not supported'
        src = body.value
        cast = src
        if (isinstance(src, tvm.expr.Cast)):
            cast = src
            src = cast.value
        # print(src)
        if (isinstance(src, tvm.expr.Add)):
            uop = 'NNPU.VAddV'
        elif (isinstance(src, tvm.expr.Sub)):
            uop = 'NNPU.VSubV'
        elif (isinstance(src, tvm.expr.Mul)):
            uop = 'NNPU.VMulV'
        elif (isinstance(src, tvm.expr.Div)):
            uop = 'NNPU.VDivV'
        elif (isinstance(src, tvm.expr.Max)):
            uop = 'NNPU.VGTMV'
        else:
            raise RuntimeError('invalid AST, unhandled op')
        lhs = src.a
        if isinstance(lhs, tvm.expr.Cast):
            lhs = lhs.value
        rhs = src.b
        if isinstance(rhs, tvm.expr.Cast):
            rhs = rhs.value
        if (not isinstance(lhs, tvm.expr.Load) or not isinstance(rhs, tvm.expr.Load)):
            raise RuntimeError('invalid AST, Load expected')
        dst_coef = tvm.arith.DetectLinearEquation(dst_idx, indices)
        lhs_coef = tvm.arith.DetectLinearEquation(lhs.index, indices)
        rhs_coef = tvm.arith.DetectLinearEquation(rhs.index, indices)
        # do stride checking
        if not util.equal_const_int(dst_coef[0], 1) or not util.equal_const_int(lhs_coef[0], 1) or not util.equal_const_int(rhs_coef[0], 1):
            raise RuntimeError('invalid AST, stride is not 1')
        
        # do type checking
        dst_type = cast.dtype
        lhs_type = lhs.dtype
        rhs_type = rhs.dtype
        if not lhs_type == rhs_type:
            raise RuntimeError('invalid AST, difference source type')
        body = tvm.call_intrin("int32", uop,
                                create_access_ptr(dst_type, dst_var, dst_coef[-1], ext, 'w'),
                                create_access_ptr(lhs_type, lhs.buffer_var, lhs_coef[-1], ext, 'r'),
                                create_access_ptr(rhs_type, rhs.buffer_var, rhs_coef[-1], ext, 'r'),
                                ext,
                                get_mode_code(lhs_type, dst_type)
                                )
        return tvm.stmt.Evaluate(body)

    def mat_vctr(stmt, params):
        # IR structure checking
        for1 = stmt
        assert isinstance(for1, tvm.stmt.For), 'invalid AST, not For loop'
        for2 = for1.body
        assert isinstance(for2, tvm.stmt.For), 'invalid AST, not For loop'
        store = for2.body
        assert isinstance(store, tvm.stmt.Store), 'invalid AST, expected Store'
        src = store.value
        if (isinstance(src, tvm.expr.Cast)):
            op = src.value
        else:
            op = src
        # check op
        if (isinstance(op, tvm.expr.Add)):
            uop = 'NNPU.MAddV'
        elif (isinstance(op, tvm.expr.Sub)):
            uop = 'NNPU.MSubV'
        elif (isinstance(op, tvm.expr.Mul)):
            uop = 'NNPU.MMulV'
        else:
            raise RuntimeError('invalid AST, unexpected Op node')
        lhs = op.a
        if (isinstance(lhs, tvm.expr.Cast)):
            lhs = lhs.value
        assert isinstance(lhs, tvm.expr.Load), 'invalid AST, expected Load'
        rhs = op.b
        if (isinstance(rhs, tvm.expr.Cast)):
            rhs = rhs.value
        assert isinstance(rhs, tvm.expr.Load), 'invalid AST, expected Load'
        
        # loop range checking
        assert util.equal_const_int(for1.min, 0), 'invalid AST, loop start is not zero'
        assert util.equal_const_int(for2.min, 0), 'invalid AST, loop start is not zero'
        shape = params['shape']
        assert util.equal_const_int(for1.extent, shape[0]), 'invalid AST, loop range dont match, {0} vs {1}'.format(for1.extent, shape[0])
        assert util.equal_const_int(for2.extent, shape[1]), 'invalid AST, loop range dont match, {0} vs {1}'.format(for2.extent, shape[1])

        # type checking
        dst_type = src.dtype
        lhs_type = lhs.dtype
        rhs_type = rhs.dtype
        if not lhs_type == rhs_type:
            raise RuntimeError('invalid AST, difference source type')

        def try_match(lhs, rhs):
            # detect index
            indices = [for1.loop_var, for2.loop_var]
            dst_coef = tvm.arith.DetectLinearEquation(store.index, indices)
            assert len(dst_coef) == 3
            assert util.equal_const_int(dst_coef[1], 1)
            lhs_coef = tvm.arith.DetectLinearEquation(lhs.index, indices)
            assert len(lhs_coef) == 3
            assert util.equal_const_int(lhs_coef[1], 1)
            rhs_coef = tvm.arith.DetectLinearEquation(rhs.index, indices)
            assert len(rhs_coef) == 3
            assert util.equal_const_int(rhs_coef[1], 1)
            if (not util.equal_const_int(rhs_coef[0], 0)):
                return None

            # build
            body = tvm.call_intrin("int32", uop,
                                create_access_ptr(dst_type, store.buffer_var, dst_coef[-1], 1, 'w'),
                                dst_coef[0] * get_dtype_bytes(dst_type),
                                create_access_ptr(lhs_type, lhs.buffer_var, lhs_coef[-1], 1, 'r'),
                                lhs_coef[0] * get_dtype_bytes(lhs_type),
                                create_access_ptr(rhs_type, rhs.buffer_var, rhs_coef[-1], 1, 'r'),
                                shape[0], shape[1],
                                get_mode_code(lhs_type, dst_type)
                                )
            return tvm.stmt.Evaluate(body)

        body = try_match(lhs, rhs)
        if (body is not None):
            return body
        else:
            body = try_match(rhs, lhs)
            assert body is not None, 'match failure, maybe not matrix-vector computation'
            return body

    def transform_(stmt):
        if (_match_pragma(stmt, 'nnpu.vector')):
            params = ast.literal_eval(stmt.value.value)
            body = None
            if (params['code'] == 'binary'):
                body = binary_(stmt.body, params)
            elif (params['code'] == 'matrix-vector'):
                body = mat_vctr(stmt.body, params)
            else:
                raise RuntimeError('unhandled type')
            # print(body)
            return body
        return None

    return tvm.ir_pass.IRTransform(
            stmt_in, transform_, None, ["AttrStmt"])

def im2col_transform(stmt):
    def transform_(stmt):
        if (_match_pragma(stmt, 'nnpu.im2col')):
            env = get_env()

            stmt = stmt.body
            # print(stmt)
            ph_loop = stmt.body
            pw_loop = ph_loop.body
            kh_loop = pw_loop.body
            kw_loop = kh_loop.body
            c_loop = kw_loop.body
            store = c_loop.body
            dst_idx = store.index
            load = store.value
            assert isinstance(load, tvm.expr.Load), 'no casting is allowed'
            src_idx = load.index
            dtype = load.dtype
            # TODO: do some checking on loop scopes
            patch_h, patch_w = util.get_const_int(ph_loop.extent), util.get_const_int(pw_loop.extent)
            kernel_h, kernel_w = util.get_const_int(kh_loop.extent), util.get_const_int(kw_loop.extent)
            src_coefs = tvm.arith.DetectLinearEquation(src_idx, [ph_loop.loop_var, pw_loop.loop_var, kh_loop.loop_var, kw_loop.loop_var])
            # TODO: do some checking!!
            stride_h, stride_w = util.get_const_int(src_coefs[0] // src_coefs[2]), util.get_const_int(src_coefs[1] // src_coefs[3])
            assert stride_h == 1 and stride_w == 1, 'only unit strides are implemented now'
            # corner_h and corner_w are the corners at two sides.
            corner_h, corner_w = kernel_h - 1, kernel_w - 1
            builder = tvm.ir_builder.create()
            with builder.for_range(stmt.min, stmt.min + stmt.extent, stmt.loop_var.name) as i1:
                # the domain of (i_patch_h + i_kernel_h) and (i_patch_w + i_kernel_w), split into two groups
                # the domain of normal ranges, in the form of [begin, end)
                normal_range_h = [corner_h, stride_h * patch_h + kernel_h - stride_h - corner_h]
                normal_range_w = [corner_w, stride_w * patch_w + kernel_w - stride_w - corner_w]
                # the domain of coners
                corner_range_h = list(range(0, corner_h))
                corner_range_h.extend(range(stride_h * patch_h + kernel_h - stride_h - corner_h, stride_h * patch_h + kernel_h - stride_h))
                corner_range_w = list(range(0, corner_w))
                corner_range_w.extend(range(stride_w * patch_w + kernel_w - stride_w - corner_w, stride_w * patch_w + kernel_w - stride_w))
                # the normal cases
                with builder.for_range(0, normal_range_h[1] - normal_range_h[0], 'ppkh') as i_ppkh_shift:
                    with builder.for_range(0, normal_range_w[1] - normal_range_w[0], 'ppkw') as i_ppkw_shift:
                        i_kh = tvm.var('chan_kernel_h')
                        i_kw = tvm.var('chan_kernel_w')
                        i_ci = tvm.var('ci')
                        i_ppkh = i_ppkh_shift + corner_h
                        i_ppkw = i_ppkw_shift + corner_w
                        ph1 = (i_ppkh - i_kh) / stride_h
                        pw1 = (i_ppkw - i_kw) / stride_w
                        # variable replacement dict
                        var_dict = {stmt.loop_var: i1, 
                                    ph_loop.loop_var: ph1, pw_loop.loop_var: pw1,
                                    kh_loop.loop_var: i_kh, kw_loop.loop_var: i_kw,
                                    c_loop.loop_var: i_ci}
                        body = tvm.ir_pass.Substitute(store, var_dict)

                        src_coefs = tvm.arith.DetectLinearEquation(body.value.index, [i_kh, i_kw, i_ci])
                        dst_coefs = tvm.arith.DetectLinearEquation(body.index, [i_kh, i_kw, i_ci])
                        assert util.equal_const_int(src_coefs[0], 0) and util.equal_const_int(src_coefs[1], 0), \
                                    'unreachable, the coefficients should be zero after substitute'
                        assert util.equal_const_int(src_coefs[2], 1), 'the last dimension should be compact'
                        assert util.equal_const_int(dst_coefs[2], 1), 'the last dimension should be compact'
                        # the extents, strides of 3 loops: kernel_h, kernel_w, inner_channel
                        extents = to_const_ints([kernel_h, kernel_w, c_loop.extent])
                        strides = to_const_ints([dst_coefs[0], dst_coefs[1], 1])
                        dst_offset = 0
                        body = tvm.call_intrin(
                                            'int32', 'NNPU.Im2Col',
                                            create_access_ptr(dtype, body.buffer_var, dst_coefs[3] + dst_offset, extents[2], 'w'),
                                            strides[0], strides[1], 
                                            create_access_ptr(dtype, body.value.buffer_var, src_coefs[3], extents[2], 'r'),
                                            extents[0], extents[1], extents[2] * get_dtype_bytes(dtype))
                        # print(body)
                        builder.emit(body)
                # the half corners
                for ppkh in corner_range_h:
                    with builder.for_range(0, normal_range_w[1] - normal_range_w[0], 'ppkw') as i_ppkw_shift:
                        # calculate the kernel scope
                        i_kh_scope = (max(0, ppkh - patch_h + 1), min(kernel_h - 1, ppkh) + 1)

                        i_ppkw = i_ppkw_shift + corner_w
                        i_kh = tvm.var('chan_kernel_h')
                        i_kw = tvm.var('chan_kernel_w')
                        i_ci = tvm.var('ci')
                        ph1 = (ppkh - i_kh) / stride_h
                        pw1 = (i_ppkw - i_kw) / stride_w
                        # variable replacement dict
                        var_dict = {stmt.loop_var: i1, 
                                    ph_loop.loop_var: ph1, pw_loop.loop_var: pw1,
                                    kh_loop.loop_var: i_kh, kw_loop.loop_var: i_kw,
                                    c_loop.loop_var: i_ci}
                        body = tvm.ir_pass.Substitute(store, var_dict)

                        src_coefs = tvm.arith.DetectLinearEquation(body.value.index, [i_kh, i_kw, i_ci])
                        dst_coefs = tvm.arith.DetectLinearEquation(body.index, [i_kh, i_kw, i_ci])
                        assert util.equal_const_int(src_coefs[0], 0) and util.equal_const_int(src_coefs[1], 0), \
                                    'unreachable, the coefficients should be zero after substitute'
                        assert util.equal_const_int(src_coefs[2], 1), 'the last dimension should be compact'
                        assert util.equal_const_int(dst_coefs[2], 1), 'the last dimension should be compact'
                        # the extents, strides of 3 loops: kernel_h, kernel_w, inner_channel
                        extents = to_const_ints([i_kh_scope[1] - i_kh_scope[0], kernel_w, c_loop.extent])
                        strides = to_const_ints([dst_coefs[0], dst_coefs[1], 1])
                        dst_offset = i_kh_scope[0] * strides[0]
                        body = tvm.call_intrin(
                                            'int32', 'NNPU.Im2Col',
                                            create_access_ptr(body.value.dtype, body.buffer_var, dst_coefs[3] + dst_offset, extents[2], 'w'),
                                            strides[0], strides[1], 
                                            create_access_ptr(body.value.dtype, body.value.buffer_var, src_coefs[3], extents[2], 'r'),
                                            extents[0], extents[1], extents[2] * get_dtype_bytes(dtype))
                        builder.emit(body)
                for ppkw in corner_range_w:
                    with builder.for_range(0, normal_range_h[1] - normal_range_h[0], 'ppkh') as i_ppkh_shift:
                        # calculate the kernel delta scope
                        i_kw_scope = (max(0, ppkw - patch_w + 1), min(kernel_w - 1, ppkw) + 1)
                        i_ppkh = i_ppkh_shift + corner_h

                        i_kh = tvm.var('chan_kernel_h')
                        i_kw = tvm.var('chan_kernel_w')
                        i_ci = tvm.var('ci')
                        ph1 = (i_ppkh - i_kh) / stride_h
                        pw1 = (ppkw - i_kw) / stride_w
                        # variable replacement dict
                        var_dict = {stmt.loop_var: i1, 
                                    ph_loop.loop_var: ph1, pw_loop.loop_var: pw1,
                                    kh_loop.loop_var: i_kh, kw_loop.loop_var: i_kw,
                                    c_loop.loop_var: i_ci}
                        body = tvm.ir_pass.Substitute(store, var_dict)

                        src_coefs = tvm.arith.DetectLinearEquation(body.value.index, [i_kh, i_kw, i_ci])
                        dst_coefs = tvm.arith.DetectLinearEquation(body.index, [i_kh, i_kw, i_ci])
                        assert util.equal_const_int(src_coefs[0], 0) and util.equal_const_int(src_coefs[1], 0), \
                                    'unreachable, the coefficients should be zero after substitute'
                        assert util.equal_const_int(src_coefs[2], 1), 'the last dimension should be compact'
                        assert util.equal_const_int(dst_coefs[2], 1), 'the last dimension should be compact'
                        # the extents, strides of 3 loops: kernel_h, kernel_w, inner_channel
                        extents = to_const_ints([kernel_h, i_kw_scope[1] - i_kw_scope[0], c_loop.extent])
                        strides = to_const_ints([dst_coefs[0], dst_coefs[1], 1])
                        dst_offset = i_kw_scope[0] * strides[1]
                        body = tvm.call_intrin(
                                            'int32', 'NNPU.Im2Col',
                                            create_access_ptr(body.value.dtype, body.buffer_var, dst_coefs[3] + dst_offset, extents[2], 'w'),
                                            strides[0], strides[1], 
                                            create_access_ptr(body.value.dtype, body.value.buffer_var, src_coefs[3], extents[2], 'r'),
                                            extents[0], extents[1], extents[2] * get_dtype_bytes(dtype))
                        builder.emit(body)
                builder2 = tvm.ir_builder.create()
                # the full corners
                for ppkh in corner_range_h:
                    for ppkw in corner_range_w:
                        # calculate the kernel delta scope
                        i_kh_scope = (max(0, ppkh - patch_h + 1), min(kernel_h - 1, ppkh) + 1)
                        i_kw_scope = (max(0, ppkw - patch_w + 1), min(kernel_w - 1, ppkw) + 1)

                        i_kh = tvm.var('chan_kernel_h')
                        i_kw = tvm.var('chan_kernel_w')
                        i_ci = tvm.var('ci')
                        ph1 = (ppkh - i_kh) / stride_h
                        pw1 = (ppkw - i_kw) / stride_w
                        # variable replacement dict
                        var_dict = {stmt.loop_var: i1, 
                                    ph_loop.loop_var: ph1, pw_loop.loop_var: pw1,
                                    kh_loop.loop_var: i_kh, kw_loop.loop_var: i_kw,
                                    c_loop.loop_var: i_ci}
                        body = tvm.ir_pass.Substitute(store, var_dict)

                        src_coefs = tvm.arith.DetectLinearEquation(body.value.index, [i_kh, i_kw, i_ci])
                        dst_coefs = tvm.arith.DetectLinearEquation(body.index, [i_kh, i_kw, i_ci])
                        assert util.equal_const_int(src_coefs[0], 0) and util.equal_const_int(src_coefs[1], 0), \
                                    'unreachable, the coefficients should be zero after substitute'
                        assert util.equal_const_int(src_coefs[2], 1), 'the last dimension should be compact'
                        assert util.equal_const_int(dst_coefs[2], 1), 'the last dimension should be compact'
                        # the extents, strides of 3 loops: kernel_h, kernel_w, inner_channel
                        extents = to_const_ints([i_kh_scope[1] - i_kh_scope[0], i_kw_scope[1] - i_kw_scope[0], c_loop.extent])
                        strides = to_const_ints([dst_coefs[0], dst_coefs[1], 1])
                        dst_offset = i_kh_scope[0] * strides[0] + i_kw_scope[0] * strides[1]
                        body = tvm.call_intrin(
                                            'int32', 'NNPU.Im2Col',
                                            create_access_ptr(body.value.dtype, body.buffer_var, dst_coefs[3] + dst_offset, extents[2], 'w'),
                                            strides[0], strides[1], 
                                            create_access_ptr(body.value.dtype, body.value.buffer_var, src_coefs[3], extents[2], 'r'),
                                            extents[0], extents[1], extents[2] * get_dtype_bytes(dtype))
                        builder2.emit(body)
                builder.emit(builder2.get())
            # the version that Canonicals loop scopes, for CPU
            # with builder.for_range(stmt.min, stmt.min + stmt.extent, stmt.loop_var.name) as i1:
            #     corner_h, corner_w = kernel_h - 1, kernel_w - 1
            #     # the corners
            #     corner_range_h = list(range(0, corner_h))
            #     corner_range_h.extend(range(stride_h * patch_h + kernel_h - stride_h - corner_h, stride_h * patch_h + kernel_h - stride_h))
            #     corner_range_w = list(range(0, corner_w))
            #     corner_range_w.extend(range(stride_w * patch_w + kernel_w - stride_w - corner_w, stride_w * patch_w + kernel_w - stride_w))
            #     for ppkh in corner_range_h:
            #         for ppkw in corner_range_w:
            #             # calculate the kernel delta scope
            #             i_kh_scope = (max(0, ppkh - patch_h + 1), min(kernel_h - 1, ppkh) + 1)
            #             i_kw_scope = (max(0, ppkw - patch_w + 1), min(kernel_w - 1, ppkw) + 1)
            #             # print(i_kh_scope, i_kw_scope)
            #             with builder.for_range(0, tvm.const(i_kh_scope[1], 'int32') - i_kh_scope[0], 'chan_kernel_h') as i_kh_shift:
            #                 with builder.for_range(0, tvm.const(i_kw_scope[1], 'int32') - i_kw_scope[0], 'chan_kernel_w') as i_kw_shift:
            #                     with builder.for_range(0, c_loop.extent, 'ci') as i_ci:
            #                         # the new patch indice
            #                         i_kh = i_kh_shift + i_kh_scope[0]
            #                         i_kw = i_kw_shift + i_kw_scope[0]
            #                         ph1 = (ppkh - i_kh) / stride_h
            #                         pw1 = (ppkw - i_kw) / stride_w

            #                         with builder.if_scope(tvm.all(ph1 >= 0, ph1 < ph_loop.extent, pw1 >= 0, pw1 < pw_loop.extent)):
            #                             # variable replacement dict
            #                             var_dict = {stmt.loop_var: i1, 
            #                                         ph_loop.loop_var: ph1, pw_loop.loop_var: pw1,
            #                                         kh_loop.loop_var: i_kh, kw_loop.loop_var: i_kw,
            #                                         c_loop.loop_var: i_ci}
            #                             body = tvm.ir_pass.Substitute(store, var_dict)
            #                             builder.emit(body)
            #     with builder.for_range(0, stride_h * ph_loop.extent + kh_loop.extent - stride_h - 2 * corner_h, 'ppkh') as i_ppkh_shift:
            #         with builder.for_range(0, stride_w * pw_loop.extent + kw_loop.extent - stride_w - 2 * corner_w, 'ppkw') as i_ppkw_shift:
            #             with builder.for_range(0, kernel_h, 'chan_kernel_h') as i_kh:
            #                 with builder.for_range(0, kernel_w, 'chan_kernel_w') as i_kw:
            #                     with builder.for_range(0, c_loop.extent, 'ci') as i_ci:
            #                         i_ppkh = i_ppkh_shift + corner_h
            #                         i_ppkw = i_ppkw_shift + corner_w
            #                         # condition, ph1 and pw1 should be integer
            #                         with builder.if_scope(tvm.all(((i_ppkh - kh_loop.loop_var) % stride_h) == 0, ((i_ppkw - kw_loop.loop_var) % stride_w) == 0)):
            #                             # the new patch indice
            #                             ph1 = (i_ppkh - i_kh) / stride_h
            #                             pw1 = (i_ppkw - i_kw) / stride_w
            #                             with builder.if_scope(tvm.all(ph1 >= 0, ph1 < ph_loop.extent, pw1 >= 0, pw1 < pw_loop.extent)):
            #                                 # variable replacement dict
            #                                 var_dict = {stmt.loop_var: i1, 
            #                                             ph_loop.loop_var: ph1, pw_loop.loop_var: pw1,
            #                                             kh_loop.loop_var: i_kh, kw_loop.loop_var: i_kw,
            #                                             c_loop.loop_var: i_ci}
            #                                 body = tvm.ir_pass.Substitute(store, var_dict)
            #                                 builder.emit(body)
            # print(builder.get())
            return builder.get()
        return None

    return tvm.ir_pass.IRTransform(
            stmt, transform_, None, ["AttrStmt"])

def remove_if_in_mkernel(stmt):
    def transform_(stmt):
        if (_match_pragma(stmt, 'remove_condition')):
            # first collect all conditions
            conditions = []
            extends = {}
            def collect_condition(stmt):
                if (isinstance(stmt, tvm.stmt.For)):
                    extends[stmt.loop_var] = util.get_const_int(stmt.extent)
                elif (isinstance(stmt, tvm.stmt.IfThenElse)):
                    likely = stmt.condition
                    assert likely.name == 'likely', 'not likely call'
                    cond = likely.args[0]
                    if (isinstance(cond, tvm.expr.LT) and isinstance(cond.a, tvm.expr.Add)):
                        # TODO: better pattern extraction.
                        ax0 = cond.a.a
                        x1 = cond.a.b
                        limit = cond.b
                        # so the condition is ax0+x1<limit
                        conditions.append((ax0, x1, cond.b))
                return None
            tvm.ir_pass.IRTransform(stmt, collect_condition, None, ['IfThenElse', 'For'])

            print(conditions)
            print(extends)

            def create_condition():
                pass

        return None
    return tvm.ir_pass.IRTransform(
            stmt, transform_, None, ["AttrStmt"])