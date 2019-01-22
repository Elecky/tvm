'''
additional ir pass for nnpu, to transform ir before lowering
'''

from .environment import get_env
import tvm
from helper import dtype_bytes as get_dtype_bytes
from topi import util
import utils

tvm_zero = tvm.const(0, 'uint32')

# some helper functions
def mark_coproc_scope(stmt):
    irb = tvm.ir_builder.create()
    irb.scope_attr(get_env().nnpu_axis, "coproc_scope", 0)
    irb.emit(stmt)
    body = irb.get()
    return body

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

def inject_dma_intrin(stmt_in):
    env = get_env()

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        #print('inject_copy called')
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

        if (src.scope == 'global' and dst.scope == env.dram_scope):
            src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)
            ndim = len(src_shape)
            # create loop vars and index
            loop_vars = []
            src_index = None
            dst_index = None
            src_pad_offset = 0
            for i in range(ndim - 1):
                var = tvm.var('i{0}'.format(i))
                #print(var)
                loop_vars.append(var)
                src_index = var * src_strides[i] if (src_index is None) else \
                            src_index + var * src_strides[i]
                #print(src_index)
                dst_index = var * dst_strides[i] if (dst_index is None) else \
                            dst_index + var * dst_strides[i]
                # inject_copy_intrin.cc modifies src_elem_offset by padding, so we modify it back
                src_pad_offset = src_pad_offset + pad_before[i] * src_strides[i] \
                                    if pad_before else \
                                 src_pad_offset
            # src_index and dst_index are index by element number
            src_index = 0 if (src_index is None) else src_index
            dst_index = 0 if (dst_index is None) else dst_index

            src_index = src_index + src.elem_offset #if src.elem_offset.defined() else \
                        #src_index
            dst_index = dst_index # access_ptr includes elem_offset already
            # NNPU_DMALoad(src_buf_addr, src_buf_offset, dst_phy_addr, dst_phy_offset, bytes)
            body = tvm.call_llvm_intrin_with_side_effect(
                            'void', "llvm.NNPU.DMALoad", tvm_zero,
                            src.data, 
                            util.simplify(src_index - src_pad_offset) * dtype_bytes,
                            dst.access_ptr('w', 'int32') + dst_index * dtype_bytes,
                            dst_shape[-1] * dtype_bytes)

            # the tvm require a stmt rather than expr, so we create a Evaluate stmt which calls body
            body = tvm.make.Evaluate(body)

            for i in reversed(range(ndim - 1)):
                # TODO: support padding here, maybe using a memset or something
                body = tvm.make.For(
                    loop_vars[i], 0 if (not pad_before) else pad_before[i], 
                    src_shape[i], 0, 0, body)  # fortype = serial
            
            body = mark_coproc_scope(body)
            return body

        elif (src.scope == env.dram_scope and dst.scope == 'global'):
            assert not (pad_after or pad_before), \
                'padding is not supported when copying to global'
            src_shape, src_strides, dst_shape, dst_strides, _, _ = \
                _fold(src.shape, src.strides, dst.shape, dst.strides)
            ndim = len(src_shape)
            # create loop vars and index
            loop_vars = []
            src_index = None
            dst_index = None
            src_pad_offset = 0
            for i in range(ndim - 1):
                var = tvm.var('i{0}'.format(i))
                loop_vars.append(var)
                src_index = var * src_strides[i] if (src_index is None) else \
                            src_index + var * src_strides[i]
                #print(src_index)
                dst_index = var * dst_strides[i] if (dst_index is None) else \
                            dst_index + var * dst_strides[i]
                # inject_copy_intrin.cc modifies src_elem_offset by padding, so we modify it back
                src_pad_offset = src_pad_offset + pad_before[i] * src_strides[i] \
                                    if pad_before else \
                                 src_pad_offset

            # src_index and dst_index are index by element number
            src_index = 0 if (src_index is None) else src_index
            dst_index = 0 if (dst_index is None) else dst_index

            src_index = src_index # access_ptr includes elem_offset already
            dst_index = dst_index + dst.elem_offset #if dst.elem_offset.defined() else \
                        #dst_index
            # NNPU_DMAStore(dst_phy_addr, dst_phy_offset, src_buf_addr, src_buf_offset, length)
            body = tvm.call_llvm_intrin_with_side_effect(
                        'void', "llvm.NNPU.DMAStore", tvm_zero,
                        dst.data, 
                        dst_index * dtype_bytes,
                        src.access_ptr('r', 'int32') + 
                            util.simplify(src_index - src_pad_offset) * dtype_bytes,
                        dst_shape[-1] * dtype_bytes)

            # the tvm require a stmt rather than expr, so we create a Evaluate stmt which calls body
            body = tvm.make.Evaluate(body)

            for i in reversed(range(ndim - 1)):
                body = tvm.make.For(
                    loop_vars[i], 0, 
                    src_shape[i], 0, 0, body)  # fortype = serial

            body = mark_coproc_scope(body)

            return body
        else:
            raise ValueError('donnot support copy from {0} to {1}'.format(
                src.scope, dst.scope
            ))
        pass

    return tvm.ir_pass.InjectCopyIntrin(stmt_in, env.dma_copy_pragma, _inject_copy)
    #src_shape = [2, 2, 16]
    #src_strides = [32, 16, 1]
    #pad_before = [2, 0, 0]
    #pad_after = [0, 0, 0]
    #
    #dst_shape = [4, 2, 16]
    #dst_strides = [32, 16, 1]
    #
    #print (_fold(src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after))

def inject_scratchpad_ls(stmt_in):
    env = get_env()

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
        
        scopes = [env.uni_scratchpad_scope, env.vctr_scratch_scope, env.mat_scratch_scope]

        if (src.scope == env.dram_scope and dst.scope in scopes):
            src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)
            ndim = len(src_shape)
            # create loop vars and index
            loop_vars = []
            src_index = None
            dst_index = None
            src_pad_offset = 0
            for i in range(ndim - 1):
                var = tvm.var('i{0}'.format(i))
                #print(var)
                loop_vars.append(var)
                src_index = var * src_strides[i] if (src_index is None) else \
                            src_index + var * src_strides[i]
                #print(src_index)
                dst_index = var * dst_strides[i] if (dst_index is None) else \
                            dst_index + var * dst_strides[i]
                # inject_copy_intrin.cc modifies src_elem_offset by padding, so we modify it back
                src_pad_offset = src_pad_offset + pad_before[i] * src_strides[i] \
                                    if pad_before else \
                                 src_pad_offset
                
            # src_index and dst_index are index by element number
            src_index = 0 if (src_index is None) else src_index
            dst_index = 0 if (dst_index is None) else dst_index

            src_index = src_index # access_ptr includes elem_offset already
            dst_index = dst_index # access_ptr includes elem_offset already
            # NNPU_ScratchpadLoad(dram_phy_addr, dram_phy_offset, dst_phy_addr, dst_phy_offset, length)
            body = tvm.call_llvm_intrin_with_side_effect(
                        'void', "llvm.NNPU.ScratchpadLoad", tvm_zero,
                        src.access_ptr('r', 'int32') +
                            util.simplify(src_index - src_pad_offset) * dtype_bytes,
                        dst.access_ptr('w', 'int32') + dst_index * dtype_bytes,
                        dst_shape[-1] * dtype_bytes)

            # the tvm require a stmt rather than expr, so we create a Evaluate stmt which calls body
            body = tvm.make.Evaluate(body)

            for i in reversed(range(ndim - 1)):
                # TODO: support padding here, maybe using a memset or something
                body = tvm.make.For(
                    loop_vars[i], 0 if (not pad_before) else pad_before[i], 
                    src_shape[i], 0, 0, body)  # fortype = serial

            body = mark_coproc_scope(body)
            
            return body

        elif (src.scope in scopes and dst.scope == env.dram_scope):
            #assert not (pad_after or pad_before), \
            #    'padding is not supported when copying to global'
            src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)
            ndim = len(src_shape)
            # create loop vars and index
            loop_vars = []
            src_index = None
            dst_index = None
            src_pad_offset = 0
            for i in range(ndim - 1):
                var = tvm.var('i{0}'.format(i))
                loop_vars.append(var)
                src_index = var * src_strides[i] if (src_index is None) else \
                            src_index + var * src_strides[i]
                #print(src_index)
                dst_index = var * dst_strides[i] if (dst_index is None) else \
                            dst_index + var * dst_strides[i]
                # inject_copy_intrin.cc modifies src_elem_offset by padding, so we modify it back
                src_pad_offset = src_pad_offset + pad_before[i] * src_strides[i] \
                                    if pad_before else \
                                 src_pad_offset
            # src_index and dst_index are index by element number
            src_index = 0 if (src_index is None) else src_index
            dst_index = 0 if (dst_index is None) else dst_index

            src_index = src_index # access_ptr includes elem_offset already
            dst_index = dst_index # access_ptr includes elem_offset already
            # NNPU_ScratchpadStore(dram_phy_addr, dram_phy_offset, src_phy_addr, src_phy_offset, length)
            #print([util.get_const_int(st) for st in src.strides])
            #print(src.data)
            body = tvm.call_llvm_intrin_with_side_effect(
                        'void', "llvm.NNPU.ScratchpadStore", tvm_zero,
                        dst.access_ptr('w', 'int32') + dst_index * dtype_bytes,
                        src.access_ptr('r', 'int32') + 
                            util.simplify(src_index - src_pad_offset) * dtype_bytes,
                        dst_shape[-1] * dtype_bytes)

            # the tvm require a stmt rather than expr, so we create a Evaluate stmt which calls body
            body = tvm.make.Evaluate(body)

            for i in reversed(range(ndim - 1)):
                # TODO: support padding here, maybe using a memset or something
                body = tvm.make.For(
                    loop_vars[i], 0 if (not pad_before) else pad_before[i], 
                    src_shape[i], 0, 0, body)  # fortype = serial

            body = mark_coproc_scope(body)

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

        if ((pad_before and not util.equal_const_int(pad_before[-1], 0)) or 
            (pad_after and not util.equal_const_int(pad_after[-1], 0))):
            raise ValueError('can not pad last dimension')
        
        assert src.dtype == dst.dtype, 'dtype of copying source and destination does not match, \
            {0} vs {1}'.format(src.dtype, dst.dtype)
        
        # check memory scope
        scopes = [env.uni_scratchpad_scope, env.vctr_scratch_scope, env.mat_scratch_scope]
        assert src.scope in scopes, 'source buffer scope is not scratchpad'
        assert dst.scope in scopes, 'dst buffer scope is not scratchpad'

        dtype_bytes = get_dtype_bytes(src.dtype)
        
        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)
        ndim = len(src_shape)

        # create loop vars and index
        loop_vars = []
        acc_indexes = [0, ]
        src_index = None
        dst_index = None
        src_pad_offset = 0
        for i in range(ndim - 1):
            var = tvm.var('i{0}'.format(i))
            loop_vars.append(var)
            src_index = var * src_strides[i] if (src_index is None) else \
                        src_index + var * src_strides[i]
            dst_index = var * dst_strides[i] if (dst_index is None) else \
                        dst_index + var * dst_strides[i]
            acc_indexes.append(dst_index)
            # inject_copy_intrin.cc modifies src_elem_offset by padding, so we modify it back
            src_pad_offset = src_pad_offset + pad_before[i] * src_strides[i] \
                                if pad_before else \
                             src_pad_offset
        # src_index and dst_index are index by element number
        src_index = 0 if (src_index is None) else src_index
        dst_index = 0 if (dst_index is None) else dst_index

        # use the last loop as inner body
        body = tvm.call_llvm_intrin_with_side_effect(
                    'void', "llvm.NNPU.ScratchpadCopy", tvm_zero,
                    dst.access_ptr('w', 'int32') + dst_index * dtype_bytes,
                    dst_strides[-1] * dtype_bytes,
                    src.access_ptr('r', 'int32') +
                        util.simplify(src_index - src_pad_offset) * dtype_bytes,
                    src_strides[-1] * dtype_bytes,
                    dtype_bytes, 
                    dst_shape[-1])
        # the tvm require a stmt rather than expr, so we create a Evaluate stmt which calls body
        body = tvm.make.Evaluate(body)

        for i in reversed(range(ndim - 1)):
            # TODO: add padding code, use memset or something.
            body = tvm.make.For(loop_vars[i], 0 if (not pad_before) else pad_before[i], 
                    src_shape[i], 0, 0, body)  # fortype = serial)
        body = mark_coproc_scope(body)
        return body
    
    return tvm.ir_pass.InjectCopyIntrin(stmt_in, env.scratchpad_copy, _inject_copy)

def inject_accTobuffer(stmt_in):
    env = get_env()

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        #print('inject_scratchpad_copy called')

        if ((pad_before and not util.equal_const_int(pad_before[-1], 0)) or 
            (pad_after and not util.equal_const_int(pad_after[-1], 0))):
            raise ValueError('can not pad last dimension')
        
        assert src.dtype == dst.dtype, 'dtype of copying source and destination does not match, \
            {0} vs {1}'.format(src.dtype, dst.dtype)
        
        # check memory scope
        scopes = [env.uni_scratchpad_scope, env.vctr_scratch_scope, env.mat_scratch_scope]
        assert src.scope == env.acc_scope, 'source scope can only be accumulation buffer'
        assert dst.scope in scopes, 'dst buffer scope is not scratchpad'

        dtype_bytes = get_dtype_bytes(src.dtype)
        
        src_shape, src_strides, dst_shape, dst_strides, pad_before, pad_after = \
                _fold(src.shape, src.strides, dst.shape, dst.strides, pad_before, pad_after)
        ndim = len(src_shape)

        # create loop vars and index
        loop_vars = []
        acc_indexes = [0, ]
        src_index = None
        dst_index = None
        src_pad_offset = 0
        for i in range(ndim - 1):
            var = tvm.var('i{0}'.format(i))
            loop_vars.append(var)
            src_index = var * src_strides[i] if (src_index is None) else \
                        src_index + var * src_strides[i]
            dst_index = var * dst_strides[i] if (dst_index is None) else \
                        dst_index + var * dst_strides[i]
            acc_indexes.append(dst_index)
            # inject_copy_intrin.cc modifies src_elem_offset by padding, so we modify it back
            src_pad_offset = src_pad_offset + pad_before[i] * src_strides[i] \
                                if pad_before else \
                             src_pad_offset
        # src_index and dst_index are index by element number
        src_index = 0 if (src_index is None) else src_index
        dst_index = 0 if (dst_index is None) else dst_index

        # use the last loop as inner body
        body = tvm.call_llvm_intrin_with_side_effect(
                    'void', "llvm.NNPU.CopyAccToBuffer", tvm_zero,
                    dst.access_ptr('w', 'int32') + dst_index * dtype_bytes,
                    dst_strides[-1] * dtype_bytes,
                    src.access_ptr('r', 'int32') +
                        util.simplify(src_index - src_pad_offset) * dtype_bytes,
                    src_strides[-1] * dtype_bytes,
                    dtype_bytes, 
                    dst_shape[-1])
        # the tvm require a stmt rather than expr, so we create a Evaluate stmt which calls body
        body = tvm.make.Evaluate(body)

        for i in reversed(range(ndim - 1)):
            # TODO: add padding code, use memset or something.
            body = tvm.make.For(loop_vars[i], 0 if (not pad_before) else pad_before[i], 
                    src_shape[i], 0, 0, body)  # fortype = serial)
        body = mark_coproc_scope(body)
        return body
    
    return tvm.ir_pass.InjectCopyIntrin(stmt_in, env.copy_acc2buf, _inject_copy)

# functions related to lift_coproc_scope ir pass starts from here.
def _is_coproc_scope_attr(op):
    return (isinstance(op, tvm.stmt.AttrStmt) and
            op.attr_key == 'coproc_scope')

def _check_coproc_scope_attr(op):
    """ if op is an 'coproc_scope' AttrStmt, return op,
        this will stop traversal down this op;
        otherwise return None, to continue traversal.
    """
    if (_is_coproc_scope_attr(op)):
        return op
    else:
        return None

def _make_coproc_scope_attr(value, body):
    env = get_env()
    node = env.nnpu_axis
    return tvm.make.AttrStmt(node, "coproc_scope", value, body)

def _lift_coproc_scope_attr(op):
    """ if every sub-node of op is 'coproc_scope' AttrStmt,
        and have same attribute value, then lift this AttrStmt;
        otherwise return None to keep original node.
    """
    # TODO: we did't check whether 'expr's can be evaluated on co-processor
    # such as extent of a For stmt.
    if (isinstance(op, tvm.stmt.For)):
        if (_is_coproc_scope_attr(op.body)):
            # print('!!!!!!! trying to replace')
            value = op.body.value
            # it is an error to use op.body = op.body.body to replace a body of 
            # one node, so we have to recreate one for node.
            body = tvm.make.For(op.loop_var, op.min, op.extent, op.for_type, 0, op.body.body)
            body = _make_coproc_scope_attr(value, body)
            return body
    elif (isinstance(op, tvm.stmt.Block)):
        if (_is_coproc_scope_attr(op.first) and
            _is_coproc_scope_attr(op.rest) and
            utils.isEqual(op.first.value, op.rest.value)):
            value = op.first.value
            body = tvm.make.Block(op.first.body, op.rest.body)
            body = _make_coproc_scope_attr(value, body)
            return body
        elif (_is_coproc_scope_attr(op.first) and
            isinstance(op.rest, tvm.stmt.Block) and
            _is_coproc_scope_attr(op.rest.first) and
            utils.isEqual(op.first.value, op.rest.first.value)):
            # this is a special case that appears in real TVM AST for NNPU.
            # we use T to indicate coproc_scope AttrStmt nodes, F as other nodes.
            # then this is the condition that a subtree is like:
            # Block(T, Block(T, F))
            # and we convert it into:
            # Block(Block(T, T), F)
            value = op.first.value
            first = tvm.make.Block(op.first.body,
                                   op.rest.first.body)
            first = _make_coproc_scope_attr(value, first)
            body = tvm.make.Block(first, op.rest.rest)
            return body
    elif (isinstance(op, tvm.stmt.ProducerConsumer)):
        if (_is_coproc_scope_attr(op.body)):
            value = op.body.value
            body = tvm.make.ProducerConsumer(op.func, op.is_producer, op.body.body)
            body = _make_coproc_scope_attr(value, body)
            return body
    elif (isinstance(op, tvm.stmt.IfThenElse)):
        if (not op.else_case.defined()):
            if (_is_coproc_scope_attr(op.then_case)):
                value = op.then_case.value
                body = tvm.make.IfThenElse(op.condition, op.then_case.body)
                body = _make_coproc_scope_attr(value, body)
                return body
        elif (_is_coproc_scope_attr(op.then_case) and
              _is_coproc_scope_attr(op.else_case) and
              utils.isEqual(op.then_case.value, op.else_case.value)):
            value = op.then_case.value
            body = tvm.make.IfThenElse(op.condition, 
                                       op.then_case.body,
                                       op.else_case.body)
            body = _make_coproc_scope_attr(value, body)
            return body

    return None
    

def lift_coproc_scope(stmt):
    stmt = tvm.ir_pass.IRTransform(stmt, _check_coproc_scope_attr, 
                                   _lift_coproc_scope_attr, [])
    return stmt