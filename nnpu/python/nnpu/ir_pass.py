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
                        src_base,
                        src_strides[-memcpy_dim:],
                        src_shape[-memcpy_dim:])
            body = tvm.make.Evaluate(body)
            return body
        else:
            irb = tvm.ir_builder.create()

            if (pad_before and util.equal_const_int(pad_before[level], 0)):
                extend = dst_strides[-1]
                # check whether axis 'level' is compact.
                l = ndim - 1
                while (l > level and dst_strides[l - 1] == dst_shape[l] * extend):
                    l = l - 1
                    extend = dst_strides[l]

                if (l == level):
                    body = create_memset(
                                dst_base, 
                                pad_before[level] * extend / dst_strides[-1], 
                                dst_strides[-1])
                    irb.emit(body)
                else:
                    raise AssertionError('can\'t pad')
                
            # iterate from 0 to src_shape[level]
            var = tvm.var('i{0}'.format(level))
            body = _build(src_base + var * src_strides[level],
                          dst_base + (var + (0 if not pad_before else pad_before[level])) \
                                     * dst_strides[level],
                          level + 1)
            loop = tvm.make.For(var, 0, src_shape[level], 0, 0, body)  # fortype = serial)
            
            irb.emit(loop)

            if (pad_after and util.equal_const_int(pad_after[level], 0)):
                extend = dst_strides[-1]
                # check whether axis 'level' is compact.
                l = ndim - 1
                while (l > level and dst_strides[l - 1] == dst_shape[l] * extend):
                    l = l - 1
                    extend = dst_strides[l]

                if (l == level):
                    body = create_memset(
                                dst_base + (src_shape[level] + pad_before[level]) * extend, 
                                pad_after[level] * extend / dst_strides[-1], 
                                dst_strides[-1])
                    irb.emit(body)
                else:
                    raise AssertionError('can\'t pad')

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