from __future__ import absolute_import
import tvm
import nnpu 
import topi
from nnpu.utils import ScheduleProcHelper
from nnvm.top import registry as reg 
import logging
from nnvm.top import nn as _nn

levels = 1
level = 16

def is_packed_layout(layout):
    """check if layout is packed layout"""
    if layout == "NHWC":
        return True
    return False

@tvm.register_func("nnvm.compiler.build_target", override=True)
def _build(funcs, target, target_host):
    tvm_t = tvm.target.create(target)
    if tvm_t.device_name == "nnpu":
        return tvm.build(funcs, target="nnpu", target_host=target_host)
    return tvm.build(funcs, target=target)
"""
@tvm.register_func("nnvm.compiler.lower", override=True)
def _lower(sch, inputs, func_name, graph):
    import traceback
    try:
        f = tvm.lower(sch, inputs, name=func_name)
        if "quantized_conv2d" in func_name:
            logging.info(graph.ir(join_entry_attrs=["shape"]))
    except Exception:
        msg = traceback.format_exc()
        msg += "Error during compile graph\n"
        msg += "--------------------------\n"
        msg += graph.ir(join_entry_attrs=["shape"])
        raise RuntimeError(msg)
    return f if isinstance(
        f, (tvm.container.Array, tuple, list)) else [f]
"""

# nnpu : relu

def compute_relu_default(data):
    """
    relu
    
    Parameters
    ------------
    data : tvm.tensor
           n-D dimension

    Returns
    ------------
    output : tvm.tensor
             n-D dimension
    """

    Imm = tvm.const(0, data.dtype)

    return tvm.compute(data.shape, lambda *i: tvm.max(data(*i), Imm), name = "relu_res", tag = "elemwise_relu")

    shape = [topi.util.get_const_int(x) for x in data.shape]
    from functools import reduce
    size = reduce(lambda x, y: x * y, shape, 1)
    assert size % 32 == 0, 'not aligned'
    print('relu')
    print(size)
    flatten_data = topi.reshape(data, (size,))
    relu = tvm.compute((size,), lambda *i: tvm.max(flatten_data(*i), Imm), name = "relu_res", tag = "elemwise_relu")
    res = topi.reshape(relu, data.shape)

    return res

@reg.register_compute("relu", level = level)
def compute_relu(attrs, inputs, out):
    return compute_relu_default(inputs[0])


def schedule_relu_default(outs):
    assert len(outs) == 1
    env = nnpu.get_env()
    output = outs[0]

    s = tvm.create_schedule(output.op)

    # ewise_ops = []
    # ewise_inputs = []
    # relu_res = []
    data = output.op.input_tensors[0]
    relu_res = s.cache_write(output, env.get_scope('buffer4'))
    flatten = s.cache_read(data, env.get_scope('buffer4'), relu_res)
    # split output
    xo, xi = s[output].split(output.op.axis[0], 256)
    tx, xo = s[output].split(xo, nparts=2)
    s[flatten].compute_at(s[output], xo)
    s[relu_res].compute_at(s[output], xo)

    s[output].bind(tx, tvm.thread_axis("cthread"))

    # pragma copy
    s[output].pragma(xi, env.dma_copy_from_buf)
    s[flatten].pragma(flatten.op.axis[0], env.dma_copy_to_buf)

    # tensorize
    xo, xi = s[relu_res].split(relu_res.op.axis[0], 32)
    s[relu_res].tensorize(xi, env.intrins.get('VGTMI', scope_in='buffer4', scope_out='buffer4', imm_value=0.0))

    # print(tvm.lower(s, [data, output], simple_mode=True))
    return s

@reg.register_schedule("relu", level = level)
def schedule_relu(attrs, outs, target):
    """
    Schedule for relu
	Parameters
	------------
	outs : Array of Tensor
	       The computation graph description of relu
		   in the format of an array of tensors
	
	Returns
	------------
	s : Schedule
	    The computation schedule for relu
    """
    target = tvm.target.create(target)
    if target.device_name == 'nnpu':
        return schedule_relu_default(outs)
    if str(target).startswith("llvm"):
        return _nn._fschedule_broadcast(attrs, outs, target)
    raise RuntimeError("not support target %s"%target)


# nnpu : dense
def compute_dense_default(data, weight, bias = None):
    """
    dense

    Parameters
    ------------
    data : tvm.Tensor
    [batch_size, in_dim]
    
    weight : tvm.Tensor
    [out_dim, in_dim]
    
    bias : tvm.Tensor, optional
    [out_dim]
    
    Returns : 
    ------------
    output : tvm.Tensor
    [batch_size, out_dim]
    """
    env = nnpu.get_env()
    assert len(data.shape) == 2 and len(weight.shape) == 2

    if bias is not None:
        assert len(bias.shape) == 1
    batch_size, in_dim = data.shape
    out_dim, _ = weight.shape
    k = tvm.reduce_axis((0, in_dim))
    print('dense')
    print(batch_size, in_dim, out_dim)
    print('=', batch_size * in_dim * out_dim)
    if bias is not None:
        first = tvm.compute((batch_size, out_dim), lambda i,j : tvm.sum(data[i, k] * weight[j, k], axis = k), name = "first", tag = "inner_dense")
        res = tvm.compute((batch_size, out_dim), lambda i,j : first[i, j] + bias[j], name = "dense_res", tag="packed_dense")
    else:
        res = tvm.compute((batch_size,out_dim), lambda i,j : tvm.sum(data[i, k] * weight[j, k], axis = k), name = "dense_res", tag = "packed_dense")
    return res

@reg.register_compute("dense", level = level)
def compute_dense(attrs, inputs, out):
	"""math: Y = XW^T + b

	"""
	if attrs["use_bias"] == "0":
		return compute_dense_default(inputs[0], inputs[1])
	elif attrs["use_bias"] == "1":
		return compute_dense_default(inputs[0], inputs[1], inputs[2])

def schedule_dense_default(attrs, outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    output = outs[0]
    ewise_ops = []
    ewise_inputs = []
    dense_res = []
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "packed_dense"
            dense_res.append(op)

    _traverse(output.op)
    ewise_ops.reverse()
    assert len(ewise_ops) == 0, 'fuse not implemented'
    
    assert len(dense_res) == 1
    gemm_shape = (8, 8, 8)
    gevm_shape = (1, 8, 8)
    factors = 16

    if attrs['use_bias'] != 0:
        vaddv_res = dense_res[0].output(0)
        gemm_res, bias = vaddv_res.op.input_tensors
        data, weight = gemm_res.op.input_tensors
    else:
        vaddv_res = None
        gemm_res = dense_res[0].output(0)
        data, weight = gemm_res.op.input_tensors
    
    N, K, M = data.shape[0], data.shape[1], weight.shape[0]
    N, K, M = [topi.util.get_const_int(x) for x in [N, K, M]]
    if (N % gemm_shape[0] == 0):
        gevm_mode = False
    else:
        gevm_mode = True
    assert K % gemm_shape[1] == 0 and M % gemm_shape[2] == 0, 'not aligned'
    if (gevm_mode):
        # cache read, cache write, and set scopes
        cdata = s.cache_read(data, env.get_scope('buffer1'), [gemm_res])
        cweight = s.cache_read(weight, env.get_scope('buffer2'), [gemm_res])
        s[gemm_res].set_scope(env.get_scope('buffer3'))
        if (vaddv_res is None):
            cout = s.cache_write(output, env.get_scope('buffer4'))
        else:
            vaddv_res = s.cache_write(output, env.get_scope('buffer4'))
            # s[vaddv_res].set_scope(env.get_scope('buffer4'))
            cbias = s.cache_read(bias, env.get_scope('buffer5'), [vaddv_res])
        # pragma read
        s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_to_buf)
        s[cweight].pragma(s[cweight].op.axis[0], env.dma_copy_to_buf)
        if (vaddv_res is not None):
            s[cbias].pragma(s[cbias].op.axis[0], env.dma_copy_to_buf)
        
        # split output
        nf, mf = 1, 512
        if (M // mf < 2):
            mf = M // 2

        no, mo, ni, mi = s[output].tile(s[output].op.axis[0], s[output].op.axis[1], nf, mf)
        mp, mo = s[output].split(mo, nparts=2)
        s[output].reorder(mp, no, mo, ni, mi)
        s[output].pragma(ni, env.dma_copy_from_buf)
        s[output].bind(mp, tvm.thread_axis("cthread"))
        
        # compute_at
        s[cbias].compute_at(s[output], mp)
        s[gemm_res].compute_at(s[output], mo)
        if (vaddv_res is not None):
            s[vaddv_res].compute_at(s[output], mo)
        else:
            s[cout].compute_at(s[output], mo)

        # split and tensorize GEMM
        ni, mi, nt, mt = s[gemm_res].tile(s[gemm_res].op.axis[0], s[gemm_res].op.axis[1], gevm_shape[0], gevm_shape[2])
        ki, kt = s[gemm_res].split(s[gemm_res].op.reduce_axis[0], 8)
        ko, ki = s[gemm_res].split(ki, 1)
        s[gemm_res].reorder(ko, ki, ni, mi, kt, nt, mt)
        s[gemm_res].tensorize(kt, env.intrins.get('GEMM', shape=gevm_shape, scope_in1='buffer1', scope_in2='buffer2', scope_out='buffer3', mode='w'))
        # compute_at data and weight
        s[cdata].compute_at(s[gemm_res], ko)
        s[cweight].compute_at(s[gemm_res], ko)

        if (vaddv_res is not None):
            # split and tensorize VAddV
            vector_unit_size = 32
            mi, mt = s[vaddv_res].split(s[vaddv_res].op.axis[1], vector_unit_size)
            s[vaddv_res].tensorize(mt, env.intrins.get('VAddV', scope_in1='buffer3', scope_in2='buffer5', scope_out='buffer4', mode='w', size=vector_unit_size))
        else:
            s[cout].pragma(s[cout].op.axis[0], env.scratchpad_copy)


        # print(tvm.lower(s, [data, weight, bias, output], simple_mode=True))

        return s

    else:
        raise NotImplementedError('gemm mode is not implemented now')

    if attrs['use_bias'] == '0':
        raise NotImplementedError('not implemented now')
        # get every stage's tensor of conv2d
        dense_stage = dense_res[0].output(0)

        # cache read data/weight to scratchpad buffer, do pragma.
        data, weight = dense_stage.op.input_tensors
        if data.dtype == dtype_n:
            modes = 'n'
        elif data.dtype == dtype_w:
            modes = 'w'
        else:
            raise RuntimeError("NPU not support dtype %s"%data.dtype)
        # cout = s.cache_write(output, env.dram_scope)
        # c2out = s.cache_write(cout, env.uni_scratchpad_scope)
        # dense_stage = s.cache_write(c2out, env.acc_scope)
        
        cdata = s.cache_read(data, env.uni_scratchpad_scope, [dense_stage])
        cweight = s.cache_read(weight, env.uni_scratchpad_scope, [dense_stage])
        
        s[cdata].pragma(s[cdata].op.axis[1], env.dma_copy_to_buf)
        s[cweight].pragma(s[cweight].op.axis[1], env.dma_copy_to_buf)
        for consumer, tensor in ewise_inputs:
            assert tensor in consumer.input_tensors
            ctensor = s.cache_read(tensor, env.uni_scratchpad_scope, [consumer])
            # s[ctensor].compute_at(s[cout], s[cout].op.axis[2])  
            s[ctensor].pragma(s[ctensor].op.axis[0], env.dma_copy_to_buf)
        # cache write output, and replace the compute stage of output with cached 'cout'
        # s[output].pragma(s[output].op.axis[1], env.dma_copy_from_buf)
        
        cout = s.cache_write(output, env.uni_scratchpad_scope)
        if (dense_stage == output):
            dense_stage = cout
        else:
            # otherwise, set_scope the conv2d_stage3 tensor to scratchpad
            s[dense_stage].set_scope(env.uni_scratchpad_scope)
            
        for idx, op in enumerate(ewise_ops):
            tensor = op.output(0)
            if (op == output):
                # if so, this compute op should be repalced with cout.
                ewise_ops[idx] = cout.op
            else:
                # otherwise, set_scope
                s[tensor].set_scope(env.uni_scratchpad_scope)
         # cache write GEMM output, and set_scope.
        dense_stage_acc = s.cache_write(dense_stage, env.acc_scope)
        s[dense_stage].set_scope(env.uni_scratchpad_scope)
        # schedule and tensorize conv2d, this may be modified in future.

        xo, xi = s[dense_stage_acc].split(s[dense_stage_acc].op.axis[1], factor = gemm_shape[2])
        ko, ki = s[dense_stage_acc].split(s[dense_stage_acc].op.reduce_axis[0], factor = gemm_shape[1])
        no, ni = s[dense_stage_acc].split(s[dense_stage_acc].op.axis[0], factor = gemm_shape[0])
        s[dense_stage_acc].reorder(no, xo, ko, ni, xi, ki)
        if modes == 'n':
            s[dense_stage_acc].tensorize(ni, env.intrins.get('GEMM', shape = gemm_shape, mode = 'inc', scope_out='acc'))
        elif modes == 'w':
            s[dense_stage_acc].tensorize(ni, env.intrins.get('GEMM', shape = gemm_shape, mode = modes, scope_out='acc'))
        
        xo, xi = s[dense_stage].split(s[dense_stage].op.axis[1], factor = gemm_shape[2])
        no, ni = s[dense_stage].split(s[dense_stage].op.axis[0], factor = gemm_shape[0])
        s[dense_stage].reorder(no, xo, ni, xi)
        s[dense_stage_acc].compute_at(s[dense_stage], xo)
        s[dense_stage].pragma(ni, env.copy_acc2buf)

        # split output.
        # to tile output, the height and width of each part we are going to compute every loop.

        oc_factor = 16
        oco, oci = s[output].split(s[output].op.axis[1], oc_factor)
        n_factor = 8
        no, ni = s[output].split(s[output].op.axis[0], n_factor)
        
        s[output].reorder(oco, no, ni, oci)
        # repragma output
        s[output].pragma(oci, env.dma_copy_from_buf)
        # compute_at data/weight load into outputs corresponding axis.
        s[cweight].compute_at(s[output], oco)
        # s[cdata].compute_at(s[output], ni)
        # # now compute_at conv2d and ewise_ops into output's corresponding axis
        s[dense_stage].compute_at(s[output], no)
        for op in ewise_ops:
            s[op].compute_at(s[output], no)
        
    else:
        dense_stage1 = dense_res[0].output(0)
        dense_stage, bias = dense_stage1.op.input_tensors
        data, weight = dense_stage.op.input_tensors
        if data.dtype == dtype_n:
            modes = 'n'
        elif data.dtype == dtype_w:
            modes = 'w'
        else:
            raise RuntimeError("NPU not support dtype %s"%data.dtype)
        s = nnpu.create_schedule(output.op)
        cout = s.cache_write(output, env.dram_scope)
        dense_stage1 = s.cache_write(cout, env.uni_scratchpad_scope)

        dense_stage_buf = dense_stage
        dense_stage = s.cache_write(dense_stage, env.acc_scope)
        cdata = s.cache_read(data, env.dram_scope, [dense_stage])
        cweight = s.cache_read(weight, env.dram_scope, [dense_stage])
        cbias = s.cache_read(bias, env.dram_scope, [dense_stage1])
        c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [dense_stage])
        c2weight = s.cache_read(cweight, env.uni_scratchpad_scope, [dense_stage])
        c2bias = s.cache_read(cbias, env.uni_scratchpad_scope, [dense_stage1])
        s[dense_stage].set_scope(env.acc_scope)
        s[dense_stage1].set_scope(env.uni_scratchpad_scope)

        s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
        s[cweight].pragma(s[cweight].op.axis[0], env.dma_copy_pragma)
        s[cbias].pragma(s[cbias].op.axis[0], env.dma_copy_pragma)
        s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls)
        s[c2weight].pragma(s[c2weight].op.axis[0], env.scratchpad_ls)
        s[c2bias].pragma(s[c2bias].op.axis[0], env.scratchpad_ls)

        s[dense_stage_buf].set_scope(env.uni_scratchpad_scope)
        s[dense_stage_buf].pragma(s[dense_stage_buf].op.axis[0], env.copy_acc2buf)
        s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
        s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)

        xo, xi = s[dense_stage].split(s[dense_stage].op.axis[1], factor = gemm_shape[2])
        ko, ki = s[dense_stage].split(s[dense_stage].op.reduce_axis[0], factor = gemm_shape[1])
        s[dense_stage].reorder(xo, ko, s[dense_stage].op.axis[0], xi, ki)
        if modes == 'n':
            s[dense_stage].tensorize(s[dense_stage].op.axis[0], env.intrins.get('GEMM', shape = gemm_shape, mode = 'inc', scope_out='acc'))
        elif modes == 'w':
            s[dense_stage].tensorize(s[dense_stage].op.axis[0], env.intrins.get('GEMM', shape = gemm_shape, mode = modes, scope_out='acc'))
        s[cdata].compute_at(s[dense_stage], ko)
        s[cweight].compute_at(s[dense_stage], ko)
        s[c2data].compute_at(s[dense_stage], ko)
        s[c2weight].compute_at(s[dense_stage], ko)

        
        xo, xi = s[dense_stage1].split(s[dense_stage1].op.axis[1], factor = factors)
        s[dense_stage1].tensorize(xi, env.intrins.get('VAddV', mode = 'w'))
        s[dense_stage_buf].compute_at(s[dense_stage1], xo)
        s[dense_stage].compute_at(s[dense_stage1], xo)
        s[cbias].compute_at(s[dense_stage1], xo)
        s[c2bias].compute_at(s[dense_stage1], xo)
    # tensorize all fused element-wise ops.
    # TODO: add tensorize of other element-wise ops
    for op in ewise_ops:
        if op.tag == "elemwise_relu":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1], 
                                env.intrins.get('VGTMI', imm_value = 0.0, mode = modes))
        elif op.tag == "elemwise_log":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                env.intrins.get('VLog', mode = modes))
        elif op.tag == "elemwise_elemwise_add":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                    env.intrins.get('VAddV', mode = modes))
            
        elif op.tag == "elemwise_elemwise_sub":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                env.intrins.get('VSubV', mode = modes))

        elif op.tag == "elemwise_elemwise_mul":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                env.intrins.get('VMulV', mode = modes))

        elif op.tag == "elemwise_exp":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                env.intrins.get('VExp', mode = modes))
        elif op.tag == "elemwise_sigmoid":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            sigmoid_stage3 = tensor.op.input_tensors[0]
            sigmoid_stage2 = sigmoid_stage3.op.input_tensors[0]
            sigmoid_stage1 = sigmoid_stage2.op.input_tensors[0]

            s[sigmoid_stage1].tensorize(s[sigmoid_stage1].leaf_iter_vars[-1], 
                                        env.intrins.get('ISubV', imm_value = 0.0, mode = modes))

            s[sigmoid_stage2].tensorize(s[sigmoid_stage2].leaf_iter_vars[-1],
                                        env.intrins.get('VExp', mode = modes))

            s[sigmoid_stage3].tensorize(s[sigmoid_stage3].leaf_iter_vars[-1],
                                        env.intrins.get('VAddI', imm_value = 1.0, mode = modes))

            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                        env.intrins.get('IDivV', imm_value = 1.0, mode = modes))
            
        else:
            raise ValueError('unhandled element-wise op')
    print(nnpu.lower(s, [data, weight, output], simple_mode = True))
    return s
	
@reg.register_schedule("dense", level = level)
def schedule_dense(attrs, outs, target):
    """
    Schedule for dense
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of dense
    	   in the format of an array of tensors

    Returns
    ------------
    s : Schedule
        The computation schedule for dense  
    """
    target = tvm.target.create(target)
    if target.device_name == "nnpu":
    	return schedule_dense_default(attrs, outs)
    if str(target).startswith("llvm"):
    	return tvm.create_schedule([x.op for x in outs])
    raise RuntimeError("not support target %s" % target)


# nnpu : elemwise_add
def compute_elemwise_add_default(lhs, rhs):
    """
    elemwise_add

    Parameters
    ------------
    lhs : tvm.Tensor
          n-D dimension

    rhs : tvm.Tensor
          n-D dimension
    Returns : 
    ------------
    output : tvm.Tensor
          n-D dimension
    """
    assert len(lhs.shape) == len(rhs.shape) 
    res = tvm.compute(lhs.shape, lambda *i : lhs(*i) + rhs(*i), name = "elemwise_add_res", tag = "elemwise_elemwise_add")
    return res

@reg.register_compute("elemwise_add", level = levels)
def compute_elemwise_add(attrs, inputs, out):
    return compute_elemwise_add_default(inputs[0], inputs[1])

def schedule_elemwise_add_default(outs):
    print("nnpu : schedule_elemwise_add")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    elemwise_add_res = []
    if output.op.input_tensors[0].dtype == output.op.input_tensors[1].dtype == dtype_n:
        modes = 'n'
    elif output.op.input_tensors[0].dtype == output.op.input_tensors[1].dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%output.op.input_tensors[0].dtype)
        """
    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "elemwise_add"
            elemwise_add_res.append(op)
    
    print(output.op)
    print(output.op.tag)
    _traverse(output.op)
    print(len(elemwise_add_res))
    """
    elemwise_add_res.append(output.op)
    assert len(elemwise_add_res) == 1
    elemwise_add_stage = elemwise_add_res[0].output(0)
    lhs, rhs = elemwise_add_stage.op.input_tensors
    factors = 16

    s = nnpu.create_schedule(output.op)

    cout = s.cache_write(output, env.dram_scope)
    elemwise_add_stage = s.cache_write(cout, env.uni_scratchpad_scope)

    clhs = s.cache_read(lhs, env.dram_scope, [elemwise_add_stage])
    crhs = s.cache_read(rhs, env.dram_scope, [elemwise_add_stage])

    c2lhs = s.cache_read(clhs, env.uni_scratchpad_scope, [elemwise_add_stage])
    c2rhs = s.cache_read(crhs, env.uni_scratchpad_scope, [elemwise_add_stage])

    s[elemwise_add_stage].set_scope(env.uni_scratchpad_scope)

    s[clhs].pragma(s[clhs].op.axis[0], env.dma_copy_pragma)
    s[crhs].pragma(s[crhs].op.axis[0], env.dma_copy_pragma)

    s[c2lhs].pragma(s[c2lhs].op.axis[0], env.scratchpad_ls)
    s[c2rhs].pragma(s[c2rhs].op.axis[0], env.scratchpad_ls)

    s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)

    lens = len(lhs.shape)
    args = [s[elemwise_add_stage].op.axis[i] for i in range(lens - 1)]
    xo, xi = s[elemwise_add_stage].split(s[elemwise_add_stage].op.axis[lens - 1], factor = factors)
    args.extend([xo, xi])
    s[elemwise_add_stage].reorder(*args)
    s[elemwise_add_stage].tensorize(xi, env.intrins.get('VAddV', mode = modes))
    return s

@reg.register_schedule("elemwise_add", level = levels)
def schedule_elemwise_add(attrs, outs, target):
    """Schedule for elemwise_add
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of elemwise_add
              in the format of an array of tensors
    Returns
    ------------
    s : Schedule
        The computation schedule for elemwise_add
    """
    target = tvm.target.create(target)
    if target.device_name == "nnpu":
    	return schedule_elemwise_add_default(outs)
    if str(target).startswith("llvm"):
    	return tvm.create_schedule([x.op for x in outs])
    raise RuntimeError("not support target %s"%target)


# nnpu : elemwise_sub
def compute_elemwise_sub_default(lhs, rhs):
    """
    elemwise_sub

    Parameters
    ------------
    lhs : tvm.Tensor
          n-D dimension
          
    rhs : tvm.Tensor
          n-D dimension
    Returns : 
    ------------
    output : tvm.Tensor
          n-D dimension
    """
    print("nnpu : compute_elemwise_sub")
    assert len(lhs.shape) == len(rhs.shape) 
    res = tvm.compute(lhs.shape, lambda *i : lhs(*i) - rhs(*i), name = "elemwise_sub_res", tag = "elemwise_elemwise_sub")
    return res

@reg.register_compute("elemwise_sub", level = levels)
def compute_elemwise_sub(attrs, inputs, out):
    return compute_elemwise_sub_default(inputs[0], inputs[1])

def schedule_elemwise_sub_default(outs):
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    elemwise_sub_res = []
    if output.op.input_tensors[0].dtype == output.op.input_tensors[1].dtype == dtype_n:
        modes = 'n'
    elif output.op.input_tensors[0].dtype == output.op.input_tensors[1].dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%output.op.input_tensors[0].dtype)
        """
    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "elemwise_add"
            elemwise_add_res.append(op)
    
    print(output.op)
    print(output.op.tag)
    _traverse(output.op)
    print(len(elemwise_add_res))
    """
    elemwise_sub_res.append(output.op)
    assert len(elemwise_sub_res) == 1
    elemwise_sub_stage = elemwise_sub_res[0].output(0)
    lhs, rhs = elemwise_sub_stage.op.input_tensors
    factors = 16

    s = nnpu.create_schedule(output.op)

    cout = s.cache_write(output, env.dram_scope)
    elemwise_sub_stage = s.cache_write(cout, env.uni_scratchpad_scope)

    clhs = s.cache_read(lhs, env.dram_scope, [elemwise_sub_stage])
    crhs = s.cache_read(rhs, env.dram_scope, [elemwise_sub_stage])

    c2lhs = s.cache_read(clhs, env.uni_scratchpad_scope, [elemwise_sub_stage])
    c2rhs = s.cache_read(crhs, env.uni_scratchpad_scope, [elemwise_sub_stage])

    s[elemwise_sub_stage].set_scope(env.uni_scratchpad_scope)

    s[clhs].pragma(s[clhs].op.axis[0], env.dma_copy_pragma)
    s[crhs].pragma(s[crhs].op.axis[0], env.dma_copy_pragma)

    s[c2lhs].pragma(s[c2lhs].op.axis[0], env.scratchpad_ls)
    s[c2rhs].pragma(s[c2rhs].op.axis[0], env.scratchpad_ls)

    s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)

    lens = len(lhs.shape)
    args = [s[elemwise_sub_stage].op.axis[i] for i in range(lens - 1)]
    xo, xi = s[elemwise_sub_stage].split(s[elemwise_sub_stage].op.axis[lens - 1], factor = factors)
    args.extend([xo, xi])
    s[elemwise_sub_stage].reorder(*args)
    s[elemwise_sub_stage].tensorize(xi, env.intrins.get('VSubV', mode = modes))
    return s

@reg.register_schedule("elemwise_sub", level = levels)
def schedule_elemwise_sub(attrs, outs, target):
    """Schedule for elemwise_sub
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of elemwise
              in the format of an array of tensors
    Returns
    ------------
    s : Schedule
        The computation schedule for elemwise_sub
    """
    target = tvm.target.create(target)
    if target.device_name == "nnpu":
    	return schedule_elemwise_sub_default(outs)
    if str(target).startswith("llvm"):
    	return tvm.create_schedule([x.op for x in outs])
    raise RuntimeError("not support target %s"%target)


# nnpu : elemwise_mul
def compute_elemwise_mul_default(lhs, rhs):
    """
    elemwise_mul

    Parameters
    ------------
    lhs : tvm.Tensor
          n-D dimension
          
    rhs : tvm.Tensor
          n-D dimension
    Returns : 
    ------------
    output : tvm.Tensor
          n-D dimension
    """
    print("nnpu : compute_elemwise_mul")
    assert len(lhs.shape) == len(rhs.shape) 
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    res = tvm.compute(lhs.shape, lambda *i : lhs(*i).astype(dtype_w) * rhs(*i).astype(dtype_w), name = "elemwise_mul_res", tag = "elemwise_elemwise_mul")
    return res

@reg.register_compute("elemwise_mul", level = levels)
def compute_elemwise_mul(attrs, inputs, out):
    return compute_elemwise_mul_default(inputs[0], inputs[1])

def schedule_elemwise_mul_default(outs):
    print("nnpu : schedule_elemwise_mul")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    elemwise_mul_res = []
    if output.op.input_tensors[0].dtype == output.op.input_tensors[1].dtype == dtype_n:
        modes = 'n'
    elif output.op.input_tensors[0].dtype == output.op.input_tensors[1].dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%output.op.input_tensors[0].dtype)

    elemwise_mul_res.append(output.op)
    assert len(elemwise_mul_res) == 1
    elemwise_mul_stage = elemwise_mul_res[0].output(0)
    lhs, rhs = elemwise_mul_stage.op.input_tensors
    factors = 16

    s = nnpu.create_schedule(output.op)

    cout = s.cache_write(output, env.dram_scope)
    elemwise_mul_stage = s.cache_write(cout, env.uni_scratchpad_scope)

    clhs = s.cache_read(lhs, env.dram_scope, [elemwise_mul_stage])
    crhs = s.cache_read(rhs, env.dram_scope, [elemwise_mul_stage])

    c2lhs = s.cache_read(clhs, env.uni_scratchpad_scope, [elemwise_mul_stage])
    c2rhs = s.cache_read(crhs, env.uni_scratchpad_scope, [elemwise_mul_stage])

    s[elemwise_mul_stage].set_scope(env.uni_scratchpad_scope)

    s[clhs].pragma(s[clhs].op.axis[0], env.dma_copy_pragma)
    s[crhs].pragma(s[crhs].op.axis[0], env.dma_copy_pragma)

    s[c2lhs].pragma(s[c2lhs].op.axis[0], env.scratchpad_ls)
    s[c2rhs].pragma(s[c2rhs].op.axis[0], env.scratchpad_ls)

    s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)

    lens = len(lhs.shape)
    args = [s[elemwise_mul_stage].op.axis[i] for i in range(lens - 1)]
    xo, xi = s[elemwise_mul_stage].split(s[elemwise_mul_stage].op.axis[lens - 1], factor = factors)
    args.extend([xo, xi])
    s[elemwise_mul_stage].reorder(*args)
    if modes == 'n':
        s[elemwise_mul_stage].tensorize(xi, env.intrins.get('VMulV', mode = 'inc'))
    elif modes == 'w':
        s[elemwise_mul_stage].tensorize(xi, env.intrins.get('VMulV', mode = 'w'))
    return s


@reg.register_schedule("elemwise_mul", level = levels)
def schedule_elemwise_mul(attrs, outs, target):
    """
    Schedule for elemwise_mul
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of elemwise_mul
              in the format of an array of tensors
    Returns
    ------------
    s : Schedule
        The computation schedule for elemwise_mul
    """
    target = tvm.target.create(target)
    if target.device_name == "nnpu":
    	return schedule_elemwise_mul_default(outs)
    if str(target).startswith("llvm"):
    	return tvm.create_schedule([x.op for x in outs])
    raise RuntimeError("not support target %s"%target)

# nnpu : exp

def compute_exp_default(input):
    """
    exp 
    Parameters
    ------------
    data : tvm.tensor
           n-D dimension

    Returns
    ------------
    output : tvm.tensor
             n-D dimension
    """
    print("nnpu : compute_exp")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    res = tvm.compute(input.shape, lambda *i : tvm.exp(input(*i).astype(dtype_w)), name = "exp_res", tag = "elemwise_exp")
    return res


def schedule_exp_default(outs):
    print("nnpu : schedule_exp")
    assert len(outs) == 1
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    output = outs[0]

    ewise_inputs = []
    ewise_ops = []
    exp_res = []
    factors = 16
    """
    def _traverse(op):
        print("op.tag : %s"%op.tag)
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                ewise_ops.append(op)
            
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    print("========")
                    _traverse(tensor.op)
                    
        else:
            assert op.tag == "elemwise"
            exp_res.append(op)
    
    print("output.op : %s"%output.op)
    _traverse(output.op)
    """
    exp_res.append(output.op)
    assert len(exp_res) == 1
    exp_stage = exp_res[0].output(0)
    data = exp_stage.op.input_tensors[0]
    if data.dtype == dtype_n:
        modes = 'n'
    elif data.dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%data.dtype)

    s = nnpu.create_schedule(output.op)
    cout = s.cache_write(output, env.dram_scope)
    exp_stage = s.cache_write(cout, env.uni_scratchpad_scope)

    cdata = s.cache_read(data, env.dram_scope, [exp_stage])
    c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [exp_stage])

    s[exp_stage].set_scope(env.uni_scratchpad_scope)

    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
    s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls)
    
    s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)

    lens = len(data.shape)

    args = [s[exp_stage].op.axis[i] for i in range(lens - 1)]
    xo, xi = s[exp_stage].split(s[exp_stage].op.axis[lens - 1], factor = factors)
    args.extend([xo, xi])
    if modes == 'n':
        s[exp_stage].tensorize(xi, env.intrins.get('VExp', mode = 'inc'))
    elif modes == 'w':
        s[exp_stage].tensorize(xi, env.intrins.get('VExp', mode = modes))
    return s

@reg.register_compute("exp", level = levels)
def compute_exp(attrs, inputs, out):
    return compute_exp_default(inputs[0])

@reg.register_schedule("exp", level = levels)
def schedule_exp(attrs, outs, target):
    """
    Schedule for exp
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of exp
              in the format of an array of tensors
    Returns
    ------------
    s : Schedule
        The computation schedule for exp
    """
    target = tvm.target.create(target)
    if target.device_name == "nnpu":
        return schedule_exp_default(outs)
    if str(target).startswith("llvm"):
        return tvm.create_schedule([x.op for x in outs])
    raise RuntimeError("not support target %s"%target)
    
# nnpu : log
def compute_log_default(data):
    """
    log compute

    Parameters
    ------------
    data : tvm.tensor
           n-D dimension

    Returns
    ------------
    output : tvm.tensor
             n-D dimension
    """
    print("nnpu : compute_log")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    res = tvm.compute(data.shape, lambda *i : tvm.log(data(*i).astype(dtype_w)), name = 'log_res', tag = 'elemwise_log')
    return res

def schedule_log_default(outs):
    print("nnpu : schedule_log")
    assert len(outs) == 1

    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    factors = 16
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    log_res = []
    assert output.dtype == dtype_w

    if output.op.input_tensors[0].dtype == dtype_n:
        modes = 'n'
    elif output.op.input_tensors[0].dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%output.op.input_tensors[0].dtype)
    
    log_res.append(output.op)
    assert len(log_res) == 1
    log_stage = log_res[0].output(0)
    data = log_stage.op.input_tensors[0]
    s = nnpu.create_schedule(output.op)

    cout = s.cache_write(output, env.dram_scope)
    log_stage = s.cache_write(cout, env.uni_scratchpad_scope)

    cdata = s.cache_read(data, env.dram_scope, [log_stage])
    c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [log_stage])

    s[log_stage].set_scope(env.uni_scratchpad_scope)

    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
    s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls)

    s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)
    lens = len(data.shape)
    args = [s[log_stage].op.axis[i] for i in range (lens - 1)]
    xo, xi = s[log_stage].split(s[log_stage].op.axis[lens - 1], factor = factors)
    args.extend([xo, xi])
    s[log_stage].reorder(*args)
    if modes == 'n':
        s[log_stage].tensorize(xi, env.intrins.get('VLog', mode = 'inc'))
    elif modes == 'w':
        s[log_stage].tensorize(xi, env.intrins.get('VLog', mode = modes))
    s[cdata].compute_at(s[log_stage], xo)
    s[c2data].compute_at(s[log_stage], xo)
    print(nnpu.lower(s, [data, output], simple_mode = True))
    return s



@reg.register_compute("log", level = levels)
def compute_log(attrs, inputs, out):
    return compute_log_default(inputs[0])


@reg.register_schedule("log", level = levels)
def schedule_log(attrs, outs, target):
    """
    Schedule for log
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of log
              in the format of an array of tensors
    Returns
    ------------
    s : Schedule
        The computation schedule for log
    """
    target = tvm.target.create(target)
    if target.device_name == "nnpu":
        return schedule_log_default(outs)
    elif str(target).startswith("llvm"):
        return tvm.create_schedule([x.op for x in outs])
    raise RuntimeError("not support target %s"%target)


def compute_softmax_default(data, axis):
    """
    softmax
	Parameters
	------------
	data : tvm.tensor
	       can be any dimension

	Returns 
	------------
	output : tvm.tensor
	         can be any dimension, same with data
    """
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    lens = len(data.shape)
    assert lens == 2
    if data.dtype == dtype_n:
        first = tvm.compute(data.shape, lambda i, j : tvm.exp(data[i, j].astype(dtype_w)), name = "first")
    else:
        first = tvm.compute(data.shape, lambda i, j : tvm.exp(data[i, j]), name = "first")
    k = tvm.reduce_axis((0, data.shape[1]))
    second = tvm.compute((data.shape[0], 1), lambda i : tvm.sum(first[i, k], axis = k), name = 'second')
    res = tvm.compute(data.shape, lambda i, j : first[i, j] / second[i, ], name = 'res', tag = 'softmax')
    return res

def schedule_softmax_default(outs):
    print("nnpu : schedule_softmax")
    env = nnpu.get_env()
    factors = 16
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    softmax_res = []
    assert output.dtype == dtype_w
    if output.op.input_tensors[0].dtype == dtype_n:
        modes = 'n'
    elif output.op.input_tensors[0].dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%output.op.input_tensors[0].dtype)
    
    softmax_res.append(output.op)
    assert len(softmax_res) == 1
    softmax_stage3 = softmax_res[0].output(0)
    softmax_stage1, softmax_stage2 = softmax_stage3.op.input_tensors
    data = softmax_stage1.op.input_tensors[0]
    s = nnpu.create_schedule(output.op)
    cout = s.cache_write(output, env.dram_scope)
    softmax_stage3 = s.cache_write(cout, env.uni_scratchpad_scope)

    s[softmax_stage3].set_scope(env.uni_scratchpad_scope)
    s[softmax_stage2].set_scope(env.uni_scratchpad_scope)
    s[softmax_stage1].set_scope(env.uni_scratchpad_scope)
    cdata = s.cache_read(data, env.dram_scope, [softmax_stage1])
    c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [softmax_stage1])

    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
    s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls)
    
    s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)
    ko, ki = s[softmax_stage1].split(s[softmax_stage1].op.axis[1], factor = factors)
    s[softmax_stage1].reorder(s[softmax_stage1].op.axis[0], ko, ki)
    if modes == 'n':
        s[softmax_stage1].tensorize(ki, env.intrins.get('VExp', mode = 'inc'))
    elif modes == 'w':
        s[softmax_stage1].tensorize(ki, env.intrins.get('VExp', mode = modes))

    s[softmax_stage2].reorder(s[softmax_stage2].op.axis[0], s[softmax_stage2].op.reduce_axis[0])

    s[softmax_stage2].tensorize(s[softmax_stage2].op.reduce_axis[0], env.intrins.get('VReduceSum', mode = 'w'))

    ko, ki = s[softmax_stage3].split(s[softmax_stage3].op.axis[1], factor = factors)
    s[softmax_stage3].tensorize(ki, env.intrins.get('VDivS', mode = 'w'))
    print(nnpu.lower(s, [data, output], simple_mode = True))
    return s


@reg.register_compute("softmax", level = levels)
def compute_softmax(attrs, inputs, out):
    axis = int(attrs['axis'])
    return compute_softmax_default(inputs[0], axis)

@reg.register_schedule("softmax", level = levels)
def schedule_softmax(attrs, outs, target):
    """
    Schedule for softmax
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of softmax
              in the format of an array of tensors
    Returns
    ------------
    s : Schedule
        The computation schedule for softmax
    """
    target = tvm.target.create(target)
    if target.device_name == "nnpu":
        return schedule_softmax_default(outs)
    if str(target).startswith("llvm"):
        return tvm.create_schedule([x.op for x in outs])
    raise RuntimeError("not support target %s"%target)


def packed_conv2d(data, kernel, bias, strides, padding, out_dtype):
    
    env = nnpu.get_env()
    assert isinstance(strides, int) or len(strides) == 2

    filter_height, filter_width, num_filter, in_channel = [topi.util.get_const_int(x) for x in kernel.shape]

    # TODO: this is for demo only
    # batch_size, in_height, in_width, in_channel = [topi.util.get_const_int(x) for x in data.shape]
    # if (in_height == 28):
    #     padding = (padding[0] + 2, padding[1] + 2)

    if isinstance(strides, int):
        stride_height = stride_width = strides
    else:
        stride_height, stride_width = strides
    out_channel = num_filter
    if(padding[0]):
        pad_data = topi.nn.pad(data, [0, padding[0], padding[1], 0], name = "pad_data")
    else:
        pad_data = data 
    batch_size, in_height, in_width, in_channel = [topi.util.get_const_int(x) for x in pad_data.shape]

    out_height = topi.util.simplify((in_height - filter_height) // stride_height + 1)
    out_height = topi.util.get_const_int(out_height)

    out_width = topi.util.simplify((in_width - filter_width) // stride_width + 1)
    out_width = topi.util.get_const_int(out_width)

    print('conv2d')
    print(out_height, out_width, num_filter, filter_height, filter_width, in_channel)
    print('=', out_height * out_width * num_filter * filter_height * filter_width * in_channel)

    k_f_h = tvm.reduce_axis((0, filter_height))
    k_f_w = tvm.reduce_axis((0, filter_width))
    k_ci_o = tvm.reduce_axis((0, in_channel // 8))
    k_ci_i = tvm.reduce_axis((0, 8))
    if (bias is None):
        col_data_shape = (batch_size, out_height, out_width, filter_height, filter_width, in_channel)
        col_data = tvm.compute(col_data_shape,
                            lambda n, h, w, kh, kw, c: pad_data[n, h + kh*stride_height, w+kw*stride_width, c],
                            name='col_data')

        assert col_data_shape[-1] % 8 == 0, 'not aligned'
        assert col_data_shape[2] % 8 == 0, 'not aligned'
        # Nz layout
        packed_data_shape = (batch_size, out_height, filter_height, filter_width, in_channel // 8, out_width, 8)
        packed_data = tvm.compute(packed_data_shape,
                            lambda n, h, kh, kw, co, w, ci: col_data[n, h, w, kh, kw, co * 8 + ci],
                            name='packed_data')
        # Nz layout
        packed_kernel_shape = (filter_height, filter_width, out_channel // 8, in_channel // 8, 8, 8)
        packed_kernel = tvm.compute(packed_kernel_shape,
                            lambda kh, kw, oco, co, oci, ci: kernel[kh, kw, oco*8+oci, co*8+ci],
                            name='packed_kernel')
        
        res1 = tvm.compute((batch_size, out_height, out_channel // 8, out_width, 8),
                        lambda n, h, oco, w, oci: 
                            tvm.sum(packed_data[n, h, k_f_h, k_f_w, k_ci_o, w, k_ci_i]
                                * packed_kernel[k_f_h, k_f_w, oco, k_ci_o, oci, k_ci_i],
                                axis=[k_f_h, k_f_w, k_ci_o, k_ci_i] ),
                        name='conv2d_inner')
        res = tvm.compute((batch_size, out_height, out_width, out_channel),
                        lambda n, h, w, oc: res1[n, h, oc / 8, w, oc % 8],
                        name='conv2d_res', tag='packed_conv2d')
    else:
        col_data_shape = (batch_size, out_height, out_width, filter_height, filter_width, in_channel)
        col_data = tvm.compute(col_data_shape,
                            lambda n, h, w, kh, kw, c: pad_data[n, h + kh*stride_height, w+kw*stride_width, c],
                            name='col_data')

        assert col_data_shape[-1] % 8 == 0, 'not aligned'
        assert col_data_shape[2] % 8 == 0, 'not aligned'
        # Nz layout
        packed_data_shape = (batch_size, out_height, filter_height, filter_width, in_channel // 8, out_width, 8)
        packed_data = tvm.compute(packed_data_shape,
                            lambda n, h, kh, kw, co, w, ci: col_data[n, h, w, kh, kw, co * 8 + ci],
                            name='packed_data')
        # Nz layout
        packed_kernel_shape = (filter_height, filter_width, out_channel // 8, in_channel // 8, 8, 8)
        packed_kernel = tvm.compute(packed_kernel_shape,
                            lambda kh, kw, oco, co, oci, ci: kernel[kh, kw, oco*8+oci, co*8+ci],
                            name='packed_kernel')
        
        res1 = tvm.compute((batch_size, out_height, out_channel // 8, out_width, 8),
                        lambda n, h, oco, w, oci: 
                            tvm.sum(packed_data[n, h, k_f_h, k_f_w, k_ci_o, w, k_ci_i]
                                * packed_kernel[k_f_h, k_f_w, oco, k_ci_o, oci, k_ci_i],
                                axis=[k_f_h, k_f_w, k_ci_o, k_ci_i] ),
                        name='conv2d_inner')
        bias_res = tvm.compute((batch_size, out_height, out_channel // 8, out_width, 8),
                        lambda n, h, oco, w, oci: res1[n, h, oco, w, oci] + bias[oco * 8 + oci],
                        name='bias_res')
        res = tvm.compute((batch_size, out_height, out_width, out_channel),
                        lambda n, h, w, oc: bias_res[n, h, oc / 8, w, oc % 8],
                        name='conv2d_res', tag='packed_conv2d')

    return res

def schedule_conv2d_default(outs, attrs):
    env = nnpu.get_env()
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        elif op.tag.startswith('inner'):
            print(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "packed_conv2d"
            conv2d_res.append(op)

    _traverse(output.op)
    ewise_ops.reverse() 

    assert len(conv2d_res) == 1
    assert len(ewise_ops) == 0, 'fuse not supported'

    N, OH, OW, OC = [topi.util.get_const_int(x) for x in output.shape]

    conv2d_res = output
    # get every stage's tensor of conv2d
    if (attrs['use_bias']):
        # bias_res = s.cache_write(output, env.get_scope('buffer4'))
        bias_res = output.op.input_tensors[0]
        conv2d_res, bias = bias_res.op.input_tensors
    else:
        # conv2d_res = s.cache_write(output, env.get_scope('buffer4'))
        conv2d_res = output.op.input_tensors[0]
        bias = None
        raise NotImplementedError('not implemented')

    packed_data, packed_kernel = conv2d_res.op.input_tensors
    col_data = packed_data.op.input_tensors[0]
    pad_data = col_data.op.input_tensors[0]
    kernel = packed_kernel.op.input_tensors[0]

    # cache read data/kernel
    if isinstance(pad_data.op, tvm.tensor.ComputeOp) and "pad" in pad_data.op.tag:
        data = pad_data.op.input_tensors[0]
    else:
        data = pad_data
        pad_data = None
    if (bias is not None):
        cbias = s.cache_read(bias, env.get_scope('buffer5'), [bias_res])

    # set scopes
    s[packed_data].set_scope(env.get_scope('buffer1'))
    s[packed_kernel].set_scope(env.get_scope('buffer2'))
    s[conv2d_res].set_scope(env.get_scope('buffer3'))
    s[bias_res].set_scope(env.get_scope('buffer4'))
    # s[cbias].set_scope(env.get_scope('buffer5'))

    # split output
    hfactor, wfactor, ocfactor = 8, 16, 128
    if (OH % (hfactor * 2) != 0):
        hfactor = 4
    # if (OW % wfactor != 0):
    #     wfactor = 8
    if (OC < ocfactor):
        ocfactor = OC
    assert OC % ocfactor == 0, 'split factor error'
    if (OH // hfactor < 2):
        hfactor = OH // 2
    print('hfactor, wfactor, ocfactor = ', hfactor, wfactor, ocfactor)

    ho, hi = s[output].split(output.op.axis[1], hfactor)
    wo, wi = s[output].split(output.op.axis[2], wfactor)
    oco, oci = s[output].split(output.op.axis[3], ocfactor)
    oci0, oci1 = s[output].split(oci, 8)  # to eliminate division
    cthread_h, ho = s[output].split(ho, nparts=2)
    s[output].reorder(cthread_h, ho, wo, oco, hi, wi, oci0, oci1)
    out_oco = oco

    s[output].bind(cthread_h, tvm.thread_axis('cthread'))

    s[cbias].compute_at(s[output], cthread_h)

    # pragma dma copy
    s[packed_data].pragma(packed_data.op.axis[0], env.dma_copy_to_buf)
    s[packed_kernel].pragma(packed_kernel.op.axis[0], env.dma_copy_to_buf)
    s[cbias].pragma(cbias.op.axis[0], env.dma_copy_to_buf)
    s[output].pragma(hi, env.dma_copy_from_buf)

    # split and tensorize bias_res
    wo, wt = s[bias_res].split(bias_res.op.axis[3], 8)
    s[bias_res].reorder(*bias_res.op.axis[0:3], wo, wt, bias_res.op.axis[4])
    # tensorize on wt
    s[bias_res].tensorize(wt, env.intrins.get('MAddV', shape=(8, 8), scope_in_mat='buffer3', scope_in_vctr='buffer5', scope_out='buffer4', mode='w'))
    s[bias_res].compute_at(s[output], out_oco)

    # split and tensorize conv2d_res
    n, h, oco, w, oci = conv2d_res.op.axis
    kfh, kfw, kcio, kcii = conv2d_res.op.reduce_axis
    wo, wt = s[conv2d_res].split(w, 8)
    in_channel_block = topi.util.get_const_int(packed_data.shape[4])
    factor = 4 if in_channel_block % 4 == 0 else 1
    kcioo, kcioi = s[conv2d_res].split(kcio, factor)
    s[conv2d_res].reorder(n, kfh, kfw, kcioo, kcioi, h, oco, wo, wt, oci, kcii)
    # tensorize on wt
    s[conv2d_res].tensorize(wt, env.intrins.get('GEMM', shape=(8, 8, 8), scope_in1='buffer1', scope_in2='buffer2', scope_out='buffer3', mode='w'))
    s[conv2d_res].compute_at(s[output], out_oco)

    # compute_at data and kernel
    s[packed_data].compute_at(s[conv2d_res], kcioo)
    s[col_data].compute_inline()
    s[packed_kernel].compute_at(s[conv2d_res], kcioo)

    # print(tvm.lower(s, [data, kernel, bias, output], simple_mode=True))
    # lib = tvm.build(s, [data, kernel, bias, output], 'nnpu', 'llvm', 'func')
    # print(lib.imported_modules[0].get_source('ir'))
    return s
    # exit()

    # cache read data/kernel to scratchpad buffer, do pragma.
    data, kernel = conv2d_stage.op.input_tensors
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None
    
    if pad_data is not None:
        cdata = pad_data
        s[pad_data].set_scope(env.dram_scope)
    else:
        cdata = s.cache_read(data, env.uni_scratchpad_scope, [conv2d_stage])

    ckernel = s.cache_read(kernel, env.uni_scratchpad_scope, [conv2d_stage])

    s[cdata].pragma(s[cdata].op.axis[3], env.dma_copy_to_buf)
    s[ckernel].pragma(s[ckernel].op.axis[2], env.dma_copy_to_buf)

    for consumer, tensor in ewise_inputs:
        assert tensor in consumer.input_tensors
        print(tensor)
        ctensor = s.cache_read(tensor, env.uni_scratchpad_scope, [consumer])
        # s[ctensor].compute_at(s[cout], s[cout].op.axis[2])  
        s[ctensor].pragma(s[ctensor].op.axis[0], env.dma_copy_to_buf)

    gemm_shape = (1, 16, 16)
    # cache write output, and replace the compute stage of output with cached 'cout'
    s[output].pragma(s[output].op.axis[3], env.dma_copy_from_buf)
    # NOTE: this may be a bug of tvm, after cache_write, cout.input_tensors are not in schedule,
    # so we store it for later use.
    output_inputs = output.op.input_tensors
    cout = s.cache_write(output, env.uni_scratchpad_scope)
    
    if (conv2d_stage == output):
        conv2d_stage = cout
    else:
        # otherwise, set_scope the conv2d_stage3 tensor to scratchpad
        s[conv2d_stage].set_scope(env.uni_scratchpad_scope)

    for idx, op in enumerate(ewise_ops):
        tensor = op.output(0)
        if (op == output):
            # if so, this compute op should be repalced with cout.
            ewise_ops[idx] = cout.op
        else:
            # otherwise, set_scope
            s[tensor].set_scope(env.uni_scratchpad_scope)

    # tensorize all fused element-wise ops.
    # TODO: add tensorize of other element-wise ops
    for op in ewise_ops:
        if op.tag == "elemwise_relu":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1], 
                                env.intrins.get('VGTMI', imm_value = 0.0, mode = modes))
        elif op.tag == "elemwise_log":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                env.intrins.get('VLog', mode = modes))
        elif op.tag == "elemwise_elemwise_add":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                    env.intrins.get('VAddV', mode = modes))
            
        elif op.tag == "elemwise_elemwise_sub":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                env.intrins.get('VSubV', mode = modes))

        elif op.tag == "elemwise_elemwise_mul":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                env.intrins.get('VMulV', mode = modes))

        elif op.tag == "elemwise_exp":
            tensor = op.output(0)
            s[tensor].split(op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                env.intrins.get('VExp', mode = modes))
        elif op.tag == "elemwise_sigmoid":
            tensor = op.output(0)
            if (op == ewise_ops[-1]):
                sigmoid_stage3 = output_inputs[0]
            else:
                sigmoid_stage3 = op.input_tensors[0]
            sigmoid_stage2 = sigmoid_stage3.op.input_tensors[0]
            sigmoid_stage1 = sigmoid_stage2.op.input_tensors[0]

            print(s[tensor].scope)
            print(sigmoid_stage3.op)
            print(s[sigmoid_stage3])

            s[sigmoid_stage1].split(s[sigmoid_stage1].op.axis[-1], factor = factors)
            s[sigmoid_stage1].set_scope(env.uni_scratchpad_scope)
            s[sigmoid_stage1].tensorize(s[sigmoid_stage1].leaf_iter_vars[-1], 
                                        env.intrins.get('ISubV', imm_value = 0.0, mode = modes))

            s[sigmoid_stage2].split(s[sigmoid_stage2].op.axis[-1], factor = factors)
            s[sigmoid_stage2].set_scope(env.uni_scratchpad_scope)
            s[sigmoid_stage2].tensorize(s[sigmoid_stage2].leaf_iter_vars[-1],
                                        env.intrins.get('VExp', mode = modes))
            s[sigmoid_stage1].compute_at(s[sigmoid_stage2],s[sigmoid_stage2].leaf_iter_vars[-2])

            s[sigmoid_stage3].split(s[sigmoid_stage3].op.axis[-1], factor = factors)
            s[sigmoid_stage3].set_scope(env.uni_scratchpad_scope)
            s[sigmoid_stage3].tensorize(s[sigmoid_stage3].leaf_iter_vars[-1],
                                        env.intrins.get('VAddI', imm_value = 1.0, mode = modes))       
            s[sigmoid_stage2].compute_at(s[sigmoid_stage3], s[sigmoid_stage3].leaf_iter_vars[-2])

            s[tensor].split(s[tensor].op.axis[-1], factor = factors)
            s[tensor].tensorize(s[tensor].leaf_iter_vars[-1],
                                        env.intrins.get('IDivV', imm_value = 1.0, mode = modes))
            s[sigmoid_stage3].compute_at(s[tensor], s[tensor].leaf_iter_vars[-2])
            
        else:
            raise ValueError('unhandled element-wise op')
    

    # cache write GEMM output, and set_scope.
    conv2d_stage_acc = s.cache_write(conv2d_stage, env.acc_scope)

    s[conv2d_stage].set_scope(env.uni_scratchpad_scope)

    # schedule and tensorize conv2d, this may be modified in future.
    oco, oci = s[conv2d_stage_acc].split(s[conv2d_stage_acc].op.axis[3], gemm_shape[2])
    no, ni = s[conv2d_stage_acc].split(s[conv2d_stage_acc].op.axis[0], gemm_shape[0])
    ico, ici = s[conv2d_stage_acc].split(s[conv2d_stage_acc].op.reduce_axis[2], gemm_shape[1])
    h, w = s[conv2d_stage_acc].op.axis[1:3]
    kh_r, kw_r = s[conv2d_stage_acc].op.reduce_axis[0:2]
    
    s[conv2d_stage_acc].reorder(no, h, w, oco, kh_r, kw_r, ico, ni, oci, ici)
    if modes == 'w':
        s[conv2d_stage_acc].tensorize(ni, env.intrins.get('GEMM', shape = gemm_shape, mode = modes, scope_out = 'acc'))    
    elif modes == 'n':
        s[conv2d_stage_acc].tensorize(ni, env.intrins.get('GEMM', shape = gemm_shape, mode = 'inc', scope_out = 'acc'))    
    
    no, ni = s[conv2d_stage].split(s[conv2d_stage].op.axis[0], gemm_shape[0])
    oco, oci = s[conv2d_stage].split(s[conv2d_stage].op.axis[3], gemm_shape[2])
    h, w = s[conv2d_stage].op.axis[1:3]
    s[conv2d_stage].reorder(no, h, w, oco, ni, oci)
    s[conv2d_stage_acc].compute_at(s[conv2d_stage], oco)

    s[conv2d_stage].pragma(ni, env.copy_acc2buf)

    # split output.
    # to tile output, the height and width of each part we are going to compute every loop.
    h_factor, w_factor = 7, 4
    print (output.shape[1], output.shape[2])
    assert int(output.shape[1]) % h_factor == 0 and int(output.shape[2]) % w_factor == 0, \
'currently, height/width of output must be divisible to h_factor and w_factor'
    ho, wo, hi, wi = s[output].tile(s[output].op.axis[1], s[output].op.axis[2], h_factor, w_factor)

    # the number of output channels every inner loop going to compute.
    oc_factor = 16
    oco, oci = s[output].split(s[output].op.axis[3], oc_factor)
    n_factor = 1
    no, ni = s[output].split(s[output].op.axis[0], n_factor)
    s[output].reorder(no, oco, ho, wo, hi, wi, ni, oci)
    # repragma output
    s[output].pragma(ni, env.dma_copy_from_buf)

    # compute_at data/kernel load into outputs corresponding axis.
    s[ckernel].compute_at(s[output], oco)
    s[cdata].compute_at(s[output], wo)

    # now compute_at conv2d and ewise_ops into output's corresponding axis
    s[conv2d_stage].compute_at(s[output], wi)
    for op in ewise_ops:
        s[op].compute_at(s[output], wi)

    # print(tvm.lower(s, [data, kernel, ewise_inputs[0][1],  output], simple_mode=True))
    return s

    
@reg.register_schedule("conv2d", level = level)
def schedule_conv2d(attrs, outs, target):
    """
    Schedule for conv2d
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of conv2d
              in the format of an array of tensors
    Returns
    ------------
    s : Schedule
        The computation schedule for conv2d
    """
    layout = attrs["layout"]
    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "nnpu":
            return schedule_conv2d_default(outs, attrs)
        if str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        raise RuntimeError("not support target %s"%target)
    return _nn.schedule_conv2d(attrs, outs, target)
    

@reg.register_compute("conv2d", level = level)
def compute_conv2d(attrs, inputs, out):
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int_tuple("groups")
    layout = attrs['layout']
    kernel_layout = attrs['kernel_layout']
    out_dtype = attrs['out_dtype']
    if (attrs['use_bias']):
        bias = inputs[2]
    else:
        bias = None
    assert dilation == (1, 1)
    assert layout == 'NHWC', 'NNPU only supports NHWC input layout.'
    assert kernel_layout == 'HWOI', 'NNPU only supports HWOI kernel layout'
    if is_packed_layout(layout):
        return packed_conv2d(inputs[0], inputs[1], bias, strides, padding, out_dtype = out_dtype)
    return _nn.compute_conv2d(attrs, inputs, out)

# nnpu : tanh

def compute_tanh_default(data):
    """
    tanh activation
    
    Parameters
    ------------
    data : tvm.tensor
           n-D dimension
           
    Returns
    ------------
    output : tvm.tensor
             n-D dimension ,same with data
    """
    print("nnpu : compute_tanh")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    if data.dtype == dtype_n:
        modes = 'n'
    elif data.dtype == dtype_w:
        modes = 'w'
    factors = 16
    env = nnpu.get_env()
    first = tvm.compute(data.shape, lambda *i : tvm.exp(data(*i).astype(dtype_w)), name = 'first')
    Imm = tvm.const(0, data.dtype)
    second = tvm.compute(data.shape, lambda *i : Imm - data(*i), name = 'second')
    
    thrid = tvm.compute(data.shape, lambda *i : tvm.exp(second(*i).astype(dtype_w)), name = 'thrid')	
    
    molecule = tvm.compute(data.shape, lambda *i : first(*i) - thrid(*i), name = 'molecule')

    Denominator = tvm.compute(data.shape, lambda *i : first(*i) + thrid(*i), name = 'Denominator')
    
    res = tvm.compute(data.shape, lambda *i : molecule(*i) / Denominator(*i), name = 'res')
    return res

def schedule_tanh_default(outs):
    print("nnpu : schedule_default")
    assert len(outs) == 1
    output = outs[0]
    env = nnpu.get_env()
    factors = 16
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    ewise_inputs = []
    ewise_ops = []
    tanh_res = []
    assert output.dtype == dtype_w
    if output.op.input_tensors[0].dtype == dtype_n:
        modes = 'n'
    elif output.op.input_tensors[0].dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%output.op.input_tensors[0].dtype)
    """
    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "tanh"
            tanh_res.append(op)
    _traverse(output.op)
    """
    tanh_res.append(output.op)
    assert len(tanh_res) == 1
    tanh_stage5 = tanh_res[0].output(0)

    tanh_stage3, tanh_stage4 = tanh_stage5.op.input_tensors
    tanh_stage0, tanh_stage2 = tanh_stage3.op.input_tensors
    tanh_stage1 = tanh_stage2.op.input_tensors[0]

    data = tanh_stage0.op.input_tensors[0]

    s = nnpu.create_schedule(output.op)

    cout = s.cache_write(output, env.dram_scope)

    tanh_stage5 = s.cache_write(cout, env.uni_scratchpad_scope)
    
     
    cdata = s.cache_read(data, env.dram_scope, [tanh_stage0])
    c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [tanh_stage0])
    s[tanh_stage0].set_scope(env.uni_scratchpad_scope)
    s[tanh_stage1].set_scope(env.uni_scratchpad_scope)
    s[tanh_stage2].set_scope(env.uni_scratchpad_scope)
    s[tanh_stage3].set_scope(env.uni_scratchpad_scope)
    s[tanh_stage4].set_scope(env.uni_scratchpad_scope)
    s[tanh_stage5].set_scope(env.uni_scratchpad_scope) 
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
    s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls) 
    
     

    s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)

    lens = len(data.shape) - 1

    xo, xi = s[tanh_stage0].split(s[tanh_stage0].op.axis[lens], factor = factors)
    if modes == 'w':
        s[tanh_stage0].tensorize(xi, env.intrins.get('VExp', mode = modes))
    elif modes == 'n':
        s[tanh_stage0].tensorize(xi, env.intrins.get('VExp', mode = 'inc'))
    Imm = tvm.const(0, data.dtype)

    xo, xi = s[tanh_stage1].split(s[tanh_stage1].op.axis[lens], factor = factors)
    s[tanh_stage1].tensorize(xi, env.intrins.get('ISubV', imm_value = Imm.value, mode = modes))
    # s[tanh_stage0].compute_at(s[tanh_stage1], xo)

    xo, xi = s[tanh_stage2].split(s[tanh_stage2].op.axis[lens], factor = factors)
    if modes == 'w':
        s[tanh_stage2].tensorize(xi, env.intrins.get('VExp', mode = modes))
    elif modes == 'n':
        s[tanh_stage2].tensorize(xi, env.intrins.get('VExp', mode = 'inc'))

    # s[tanh_stage1].compute_at(s[tanh_stage2], xo)
    if lens == 0:
        xo, xi = s[tanh_stage3].split(s[tanh_stage3].op.axis[lens], factor = factors)
        s[tanh_stage3].tensorize(xi, env.intrins.get('VSubV', mode = 'w'))
        # s[tanh_stage2].compute_at(s[tanh_stage3], xo)
        xo, xi = s[tanh_stage4].split(s[tanh_stage4].op.axis[lens], factor = factors)
        s[tanh_stage4].tensorize(xi, env.intrins.get('VAddV', mode = 'w'))
        # s[tanh_stage3].compute_at(s[tanh_stage4], xo)
    else:
        xo, xi = s[tanh_stage3].split(s[tanh_stage3].op.axis[lens], factor = factors)
        yo, yi = s[tanh_stage3].split(s[tanh_stage3].op.axis[lens - 1], factor = factors)
        args = [s[tanh_stage3].op.axis[i] for i in range(lens - 1)]
        args.extend([xo, yo, xi, yi])
        s[tanh_stage3].reorder(*args)
        s[tanh_stage3].tensorize(xi, env.intrins.get('MSubM', mode = 'w'))
        # s[tanh_stage2].compute_at(s[tanh_stage3], yo)

        xo, xi = s[tanh_stage4].split(s[tanh_stage4].op.axis[lens], factor = factors)
        yo, yi = s[tanh_stage4].split(s[tanh_stage4].op.axis[lens - 1], factor = factors)
        args = [s[tanh_stage4].op.axis[i] for i in range(lens - 1)]
        args.extend([xo, yo, xi, yi])
        s[tanh_stage4].reorder(*args)
        s[tanh_stage4].tensorize(xi, env.intrins.get('MAddM', mode = 'w'))
        # s[tanh_stage3].compute_at(s[tanh_stage4], yo)
    xo, xi = s[tanh_stage5].split(s[tanh_stage5].op.axis[lens], factor = factors)
    s[tanh_stage5].tensorize(xi, env.intrins.get('VDivV', mode = 'w'))
    # s[tanh_stage4].compute_at(s[tanh_stage5], xo)
    return s
    

@reg.register_compute("tanh", level = levels)
def compute_tanh(attrs, inputs, out):
    return compute_tanh_default(inputs[0])

@reg.register_schedule("tanh", level = levels)
def schedule_tanh(attrs, outs, target):
    """
    Schedule for tanh
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of tanh
              in the format of an array of tensors
    Returns
    ------------
    s : Schedule
        The computation schedule for tanh
    """
    target = tvm.target.create(target)
    if target.device_name == 'nnpu':
        return schedule_tanh_default(outs)
    if str(target).startswith("llvm"):
        return tvm.create_schedule([x.op for x in outs])
    raise RuntimeError("not support target %s"%target)

# nnpu : sigmoid

def compute_sigmoid_default(data):
    """
    sigmoid activation

    Parameters
    ------------
    data : tvm.tensor
           n-D dimension

    Returns
    ------------
    output : tvm.tensor
             n-D dimension
    """
    print("nnpu : compute_sigmoid")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    if data.dtype == dtype_n:
        modes = 'n'
    elif data.dtype == dtype_w:
        modes = 'w'
    Imm = tvm.const(0, data.dtype)
    first = tvm.compute(data.shape, lambda *i : Imm - data(*i), name = 'inner_first', tag = "inner_first")

    second = tvm.compute(data.shape, lambda *i : tvm.exp(first(*i).astype(dtype_w)), name = 'inner_second', tag = "inner_second")

    Imm_1 = tvm.const(1, dtype_w)

    thrid = tvm.compute(data.shape, lambda *i : second(*i) + Imm_1, name = 'inner_thrid', tag = "inner_thrid")

    res = tvm.compute(data.shape, lambda *i : Imm_1 / thrid(*i), name = 'sigmoid_res', tag = "elemwise_sigmoid")

    return res

def schedule_sigmoid_default(outs):
    print("nnpu : schedule_sigmoid")
    assert len(outs) == 1
    output = outs[0]
    env = nnpu.get_env()
    factors = 16
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    
    sigmoid_res = []
    sigmoid_res.append(output.op)
    sigmoid_stage3 = sigmoid_res[0].output(0)
    
    sigmoid_stage2 = sigmoid_stage3.op.input_tensors[0]

    sigmoid_stage1 = sigmoid_stage2.op.input_tensors[0]

    sigmoid_stage = sigmoid_stage1.op.input_tensors[0]

    data = sigmoid_stage.op.input_tensors[0]

    if data.dtype == dtype_n:
        modes = 'n'
    elif data.dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%data.dtype)
    s = nnpu.create_schedule(output.op)
    cout = s.cache_write(output, env.dram_scope)

    sigmoid_stage3 = s.cache_write(cout, env.uni_scratchpad_scope)

    cdata = s.cache_read(data, env.dram_scope, [sigmoid_stage])
    c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [sigmoid_stage])

    s[sigmoid_stage].set_scope(env.uni_scratchpad_scope)
    s[sigmoid_stage1].set_scope(env.uni_scratchpad_scope)
    s[sigmoid_stage2].set_scope(env.uni_scratchpad_scope)
    s[sigmoid_stage3].set_scope(env.uni_scratchpad_scope)

    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
    s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls)

    s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)
    Imm = tvm.const(0, data.dtype)
    lens = len(data.shape) - 1
    xo, xi = s[sigmoid_stage].split(s[sigmoid_stage].op.axis[lens], factor = factors)
    s[sigmoid_stage].tensorize(xi, env.intrins.get('ISubV', imm_value = Imm.value, mode = modes))
    s[cdata].compute_at(s[sigmoid_stage], xo)
    s[c2data].compute_at(s[sigmoid_stage], xo)
    xo, xi = s[sigmoid_stage1].split(s[sigmoid_stage1].op.axis[lens], factor = factors)
    if modes == 'n':
        s[sigmoid_stage1].tensorize(xi, env.intrins.get('VExp', mode = 'inc'))
    elif modes == 'w':
        s[sigmoid_stage1].tensorize(xi, env.intrins.get('VExp', mode = modes))
    s[sigmoid_stage].compute_at(s[sigmoid_stage1], xo)
    Imm_1 = tvm.const(1, dtype_w)

    xo, xi = s[sigmoid_stage2].split(s[sigmoid_stage2].op.axis[lens], factor = factors)
    s[sigmoid_stage2].tensorize(xi, env.intrins.get('VAddI', imm_value = Imm_1.value, mode = 'w'))
    s[sigmoid_stage1].compute_at(s[sigmoid_stage2], xo)

    xo, xi = s[sigmoid_stage3].split(s[sigmoid_stage3].op.axis[lens], factor = factors)
    s[sigmoid_stage3].tensorize(xi, env.intrins.get('IDivV', imm_value = Imm_1.value, mode = 'w'))
    s[sigmoid_stage2].compute_at(s[sigmoid_stage3], xo)
    # s[sigmoid_stage3].compute_at(s[cout], s[cout].op.axis[1])
    print(nnpu.lower(s, [data], simple_mode = True))
    return s


@reg.register_compute("sigmoid", level = levels)
def compute_sigmoid(attrs, inputs, out):
	return compute_sigmoid_default(inputs[0])
    
    
@reg.register_schedule("sigmoid", level = levels)
def schedule_sigmoid(attrs, outs, target):
    """
    Schedule for sigmoid
    Parameters
    ------------
    outs : Array of Tensor
           The computation graph description of sigmoid
		   in the format of an array of tensors
	
	Returns
	------------
	s : Schedule
	    The computation schedule for sigmoid
    """
    target = tvm.target.create(target)
    if target.device_name == 'nnpu':
        return schedule_sigmoid_default(outs)
    if str(target).startswith("llvm"):
        return tvm.create_schedule([x.op for x in range(outs)])
    raise RuntimeError("not support target %s"%target)

# nnpu : max_pool2d
def compute_max_pool2d_default(data, pool_size, strides, padding):
    env = nnpu.get_env()
    print("nnpu : max_pool2 compute")
    
    assert isinstance(strides, int) or len(strides) == 2
    assert len(pool_size) == 2
    if isinstance(strides, int):
        stride_height = stride_width = strides
    else:
        stride_height, stride_width = strides
    
    pool_height, pool_width = pool_size
    if(padding[0]):
        pad_data = topi.nn.pad(data, [0, padding[0], padding[1], 0], name = "pad_data")
    else:
        pad_data = data
    batch_size, in_height, in_width, channel = pad_data.shape
    out_height = topi.util.simplify((in_height - pool_height) // stride_height + 1)
    out_width = topi.util.simplify((in_width - pool_width) // stride_width + 1)
    print(batch_size, out_height, out_width, pool_height, pool_width, channel)
    print('=', batch_size * out_height * out_width * pool_height * pool_width * channel)

    p_h = tvm.reduce_axis((0, pool_height))
    p_w = tvm.reduce_axis((0, pool_width))

    res = tvm.compute((batch_size, out_height, out_width, channel),
                    lambda b_c, o_h, o_w, c : tvm.max(pad_data[b_c, o_h * stride_height + p_h, o_w*stride_width+p_w, c], axis=[p_h, p_w]), 
                    name = 'res', tag = "max_pool2d")
    return res


@reg.register_compute("max_pool2d", level = level)
def compute_max_pool2d(attrs, inputs, out):
    layout = attrs['layout']
    data = inputs[0]
    pool_size = attrs.get_int_tuple("pool_size")
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    if is_packed_layout(layout):
        return compute_max_pool2d_default(data, pool_size, strides, padding)

def schedule_max_pool2d_default(outs):
    env = nnpu.get_env()
    
    assert len(outs) == 1
    
    output = outs[0]
    data = output.op.input_tensors[0]
    
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None
    
    s = tvm.create_schedule(output.op)
    # cache write
    cout = s.cache_write(output, env.get_scope('buffer4'))
    
    if pad_data is not None:
        cdata = pad_data
        s[cdata].set_scope(env.get_scope('buffer5'))
    else:
        cdata = s.cache_read(data, env.get_scope('buffer5'), [cout])
    # pragma
    s[cdata].pragma(cdata.op.axis[0], env.dma_copy_to_buf)
    
    N, H, W, C = [topi.util.get_const_int(x) for x in output.shape]
    hfactor, wfactor, cfactor = 4, 4, 4*32
    hfactor = hfactor if hfactor <= H else H
    wfactor = wfactor if wfactor <= W else W
    cfactor = cfactor if cfactor <= C else C

    # split and tensorize
    n, h, w, c = s[cout].op.axis
    co, ci = s[cout].split(c, 32)
    ph, pw = s[cout].op.reduce_axis
    pw, dummy = s[cout].split(pw, 1)
    s[cout].reorder(n, ph, pw, h, w, co, dummy, ci)
    s[cout].tensorize(dummy, env.intrins.get('VGTMMerge', scope_in='buffer5', scope_out='buffer4', mode='w', nDim=2, size=32))

    # split output
    n, h, w, c = output.op.axis
    ho, wo, hi, wi = s[output].tile(h, w, hfactor, wfactor)
    co, ci = s[output].split(c, cfactor)
    s[output].reorder(n, ho, wo, co, hi, wi, ci)
    s[output].pragma(hi, env.dma_copy_from_buf)

    # compute_at
    s[cout].compute_at(s[output], co)
    s[cdata].compute_at(s[output], co)

    return s

    
@reg.register_schedule("max_pool2d", level = level)
def schedule_max_pool2d(attrs, outs, target):
    layout = attrs["layout"]
    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "nnpu":
            return schedule_max_pool2d_default(outs)
        if str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        raise RuntimeError("not support target %s"%target)
    return _nn.schedule_max_pool2d(attrs, outs, target)

# nnpu : global_max_pool2d
def compute_global_max_pool2d_default(data):
    print("nnpu : global_max_pool2 compute")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    
    batch_size, in_height, in_width, channel = data.shape

    k_k_h = tvm.reduce_axis((0, in_height))
    k_k_w = tvm.reduce_axis((0, in_width))

    first = tvm.compute((batch_size,in_height, 1, channel), 
                                lambda b_c, i_h, o_w, c : tvm.max(data[b_c, i_h, o_w + k_k_w, c], axis = k_k_w), name = "first")
    res = tvm.compute((batch_size, 1, 1, channel),
                                lambda b_c, o_h, o_w, c : tvm.max(first[b_c, o_h + k_k_h, o_w, c], axis = k_k_h), name = 'res', tag = "global_max_pool2d")
    return res
    

@reg.register_compute("global_max_pool2d", level = levels)
def compute_global_max_pool2d(attrs, inputs, out):
    layout = attrs['layout']
    data = inputs[0]
    if is_packed_layout(layout):
        return compute_global_max_pool2d_default(data)
    return _nn.compute_global_max_pool2d(attrs, inputs, out)

def schedule_global_max_pool2d_default(outs):
    print("nnpu : global_max_pool2 schedule")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    assert len(outs) == 1
    factors = 16
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    max_pool2d_res = []
    max_pool2d_res.append(output.op)
    assert len(max_pool2d_res) == 1
    max_pool2d_stage1 = max_pool2d_res[0].output(0)
    max_pool2d_stage = max_pool2d_stage1.op.input_tensors[0]
    data = max_pool2d_stage.op.input_tensors[0]
    if data.dtype == dtype_n:
        modes = 'n'
    elif data.dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%data.dtype)
    s = nnpu.create_schedule(output.op)
    cout = s.cache_write(output, env.dram_scope)
    max_pool2d_stage1 = s.cache_write(cout, env.uni_scratchpad_scope)
    cdata = s.cache_read(data, env.dram_scope, [max_pool2d_stage])
    c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [max_pool2d_stage])
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
    s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls)

    s[max_pool2d_stage].set_scope(env.uni_scratchpad_scope)
    s[max_pool2d_stage1].set_scope(env.uni_scratchpad_scope)
    s[cout].pragma(s[cout].op.axis[3], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[3], env.dma_copy_pragma)

    ko, ki = s[max_pool2d_stage].split(s[max_pool2d_stage].op.reduce_axis[0], factor = 1)
    xo, xi = s[max_pool2d_stage].split(s[max_pool2d_stage].op.axis[3], factor = factors)
    m, l, n = s[max_pool2d_stage].op.axis[0:3]
    s[max_pool2d_stage].reorder(m, l, n, xo, ko, ki, xi)
    s[max_pool2d_stage].tensorize(ki, env.intrins.get('VGTMMerge', mode = modes))
    s[cdata].compute_at(s[max_pool2d_stage], ko)
    s[c2data].compute_at(s[max_pool2d_stage], ko)

    ko, ki = s[max_pool2d_stage1].split(s[max_pool2d_stage1].op.reduce_axis[0], factor=1)
    xo, xi = s[max_pool2d_stage1].split(s[max_pool2d_stage1].op.axis[3], factor=16)
    m, l, n = s[max_pool2d_stage1].op.axis[0:3]
    s[max_pool2d_stage1].reorder(m, l, n, xo, ko, ki, xi)
    s[max_pool2d_stage1].tensorize(ki, env.intrins.get('VGTMMerge', mode = modes, nDim = 3))
    s[max_pool2d_stage].compute_at(s[max_pool2d_stage1], ko)
    s[max_pool2d_stage1].compute_at(s[cout], s[cout].op.axis[2])
    print(nnpu.lower(s, [data], simple_mode = True))
    return s

    
@reg.register_schedule("global_max_pool2d", level = levels)
def schedule_global_max_pool2d(attrs, outs, target):
    layout = attrs["layout"]
    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "nnpu":
            return schedule_global_max_pool2d_default(outs)
        if str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        raise RuntimeError("not support target %s"%target)
    return _nn.schedule_global_max_pool2d(attrs, outs, target)
    
# nnpu : avg_pool2d
def compute_avg_pool2d_default(data, pool_size, strides, padding):
    print("nnpu : avg_pool2d compute")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    assert isinstance(strides, int) or len(strides) == 2
    assert len(pool_size) == 2
    if isinstance(strides, int):
        stride_height = stride_width = strides
    else:
        stride_height, stride_width = strides
    
    kernel_height, kernel_width = pool_size
    if(padding[0]):
        pad_data = topi.nn.pad(data, [0, padding[0], padding[1], 0], name = "pad_data")
    else:
        pad_data = data
    batch_size, in_height, in_width, channel = pad_data.shape
    out_height = topi.util.simplify((in_height - kernel_height) // stride_height + 1)
    out_width = topi.util.simplify((in_width - kernel_width) // stride_width + 1)

    k_k_h = tvm.reduce_axis((0, kernel_height))
    k_k_w = tvm.reduce_axis((0, kernel_width))

    first = tvm.compute((batch_size,in_height, out_width, channel), 
                                lambda b_c, i_h, o_w, c : tvm.sum(pad_data[b_c, i_h, o_w * stride_width + k_k_w, c], axis = k_k_w), name = "first")
    second = tvm.compute((batch_size, out_height, out_width, channel),
                                lambda b_c, o_h, o_w, c : tvm.sum(first[b_c, o_h * stride_height + k_k_h, o_w, c], axis = k_k_h), name = "second")

    Imm = tvm.const(kernel_height * kernel_width, env.cfg['dtype_w'])

    res = tvm.compute((batch_size, out_height, out_width, channel), 
                                lambda b_c, o_h, o_w, c : second[b_c, o_h, o_w, c].astype(dtype_w) / Imm, name = "res", tag = "avg_pool")
    return res
    

@reg.register_compute("avg_pool2d", level = levels)
def compute_avg_pool2d(attrs, inputs, out):
    layout = attrs['layout']
    pool_size = attrs.get_int_tuple("pool_size")
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    data = inputs[0]
    if is_packed_layout(layout):
        return compute_avg_pool2d_default(data, pool_size, strides, padding)
    return _nn.compute_avg_pool2d(attrs, inputs, out)

def schedule_avg_pool2d_default(attrs, outs):
    print("nnpu : avg_pool2d schedule")
    pool_size = attrs.get_int_tuple("pool_size")
    
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    Imm = tvm.const(pool_size[0] * pool_size[1], dtype_w)
    assert len(outs) == 1
    factors = 16
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    avg_pool2d_res = []
    avg_pool2d_res.append(output.op)
    assert len(avg_pool2d_res) == 1
    avg_pool2d_stage2 = avg_pool2d_res[0].output(0)
    avg_pool2d_stage1 = avg_pool2d_stage2.op.input_tensors[0]
    avg_pool2d_stage = avg_pool2d_stage1.op.input_tensors[0]
    data = avg_pool2d_stage.op.input_tensors[0]
    if data.dtype == dtype_n:
        modes = 'n'
    elif data.dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%data.dtype)
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None
    
    s = nnpu.create_schedule(output.op)
    cout = s.cache_write(output, env.dram_scope)
    avg_pool2d_stage2 = s.cache_write(cout, env.uni_scratchpad_scope)
    if pad_data is not None:
        cdata = pad_data
        s[pad_data].set_scope(env.dram_scope)
    else:
        cdata = s.cache_read(data, env.dram_scope, [avg_pool2d_stage])
    c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [avg_pool2d_stage])
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
    s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls)
    s[avg_pool2d_stage].set_scope(env.uni_scratchpad_scope)
    s[avg_pool2d_stage1].set_scope(env.uni_scratchpad_scope)
    s[avg_pool2d_stage2].set_scope(env.uni_scratchpad_scope)
    s[cout].pragma(s[cout].op.axis[3], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[3], env.dma_copy_pragma)


    ko, ki = s[avg_pool2d_stage].split(s[avg_pool2d_stage].op.reduce_axis[0], factor = 1)
    xo, xi = s[avg_pool2d_stage].split(s[avg_pool2d_stage].op.axis[3], factor = factors)
    m, l, n = s[avg_pool2d_stage].op.axis[0:3]
    s[avg_pool2d_stage].reorder(m, l, n, xo, ko, ki, xi)
    s[avg_pool2d_stage].tensorize(ki, env.intrins.get('VAddMerge', mode = modes))

    s[cdata].compute_at(s[avg_pool2d_stage], ko)
    s[c2data].compute_at(s[avg_pool2d_stage], ko)
    ko, ki = s[avg_pool2d_stage1].split(s[avg_pool2d_stage1].op.reduce_axis[0], factor=1)
    xo, xi = s[avg_pool2d_stage1].split(s[avg_pool2d_stage1].op.axis[3], factor = factors)
    m, l, n = s[avg_pool2d_stage1].op.axis[0:3]
    s[avg_pool2d_stage1].reorder(m, l, n, xo, ko, ki, xi)
    s[avg_pool2d_stage1].tensorize(ki, env.intrins.get('VAddMerge', mode = modes, nDim = 3))

    s[avg_pool2d_stage].compute_at(s[avg_pool2d_stage1], ko)
    xo, xi = s[avg_pool2d_stage2].split(s[avg_pool2d_stage2].op.axis[3], factor = factors)
    m, l, n = s[avg_pool2d_stage2].op.axis[0:3]
    s[avg_pool2d_stage2].reorder(m, l, n, xo, xi)
    if modes == 'w':
        s[avg_pool2d_stage2].tensorize(xi, env.intrins.get('VDivI', imm_value = Imm.value, mode = modes))
    elif modes == 'n':
        s[avg_pool2d_stage2].tensorize(xi, env.intrins.get('VDivI', imm_value = Imm.value, mode = 'inc'))
    s[avg_pool2d_stage1].compute_at(s[avg_pool2d_stage2], xo)
    s[avg_pool2d_stage2].compute_at(s[cout], s[cout].op.axis[2])
    print(nnpu.lower(s, [data, output], simple_mode = True))
    return s
    
@reg.register_schedule("avg_pool2d", level = levels)
def schedule_avg_pool2d(attrs, outs, target):
    layout = attrs["layout"]
    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "nnpu":
            return schedule_avg_pool2d_default(attrs, outs)
        if str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        raise RuntimeError("not support target %s"%target)
    return _nn.schedule_avg_pool2d(attrs, outs, target)
    
# nnpu : global_avg_pool2d
def compute_global_avg_pool2d_default(data):
    print("nnpu : compute_global_avg_pool2d")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    batch_size, in_height, in_width, channel = data.shape

    k_k_h = tvm.reduce_axis((0, in_height))
    k_k_w = tvm.reduce_axis((0, in_width))

    first = tvm.compute((batch_size,in_height, 1, channel), 
                                lambda b_c, i_h, o_w, c : tvm.sum(data[b_c, i_h, o_w + k_k_w, c], axis = k_k_w), name = "first")
    second = tvm.compute((batch_size, 1, 1, channel),
                                lambda b_c, o_h, o_w, c : tvm.sum(first[b_c, o_h + k_k_h, o_w, c], axis = k_k_h), name = "second")

    Imm = tvm.const(topi.util.get_const_int(in_height) * topi.util.get_const_int(in_width), env.cfg['dtype_w'])
    res = tvm.compute((batch_size, 1, 1, channel), 
                                lambda b_c, o_h, o_w, c : second[b_c, o_h, o_w, c].astype(dtype_w) / Imm, name = "res", tag = "avg_pool")
    return res
    

@reg.register_compute("global_avg_pool2d", level = levels)
def compute_global_avg_pool2d(attrs, inputs, out):
    layout = attrs['layout']
    data = inputs[0]
    if is_packed_layout(layout):
        return compute_global_avg_pool2d_default(data)
    return _nn.compute_avg_pool2d(attrs, inputs, out)

def schedule_global_avg_pool2d_default(attrs, outs):
    print("nnpu : schedule_global_avg_pool2d")
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    
    assert len(outs) == 1
    factors = 16
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    avg_pool2d_res = []
    avg_pool2d_res.append(output.op)
    assert len(avg_pool2d_res) == 1
    avg_pool2d_stage2 = avg_pool2d_res[0].output(0)
    avg_pool2d_stage1 = avg_pool2d_stage2.op.input_tensors[0]
    avg_pool2d_stage = avg_pool2d_stage1.op.input_tensors[0]
    data = avg_pool2d_stage.op.input_tensors[0]
    pool_size = data.shape
    Imm = tvm.const(topi.util.get_const_int(pool_size[1] * pool_size[2]), dtype_w)
    if data.dtype == dtype_n:
        modes = 'n'
    elif data.dtype == dtype_w:
        modes = 'w'
    else:
        raise RuntimeError("NPU not support dtype %s"%data.dtype)
    s = nnpu.create_schedule(output.op)
    cout = s.cache_write(output, env.dram_scope)
    avg_pool2d_stage2 = s.cache_write(cout, env.uni_scratchpad_scope)
    
    cdata = s.cache_read(data, env.dram_scope, [avg_pool2d_stage])
    c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [avg_pool2d_stage])
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
    s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls)
    s[avg_pool2d_stage].set_scope(env.uni_scratchpad_scope)
    s[avg_pool2d_stage1].set_scope(env.uni_scratchpad_scope)
    s[avg_pool2d_stage2].set_scope(env.uni_scratchpad_scope)
    s[cout].pragma(s[cout].op.axis[3], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[3], env.dma_copy_pragma)

    ko, ki = s[avg_pool2d_stage].split(s[avg_pool2d_stage].op.reduce_axis[0], factor = 1)
    xo, xi = s[avg_pool2d_stage].split(s[avg_pool2d_stage].op.axis[3], factor = factors)
    m, l, n = s[avg_pool2d_stage].op.axis[0:3]
    s[avg_pool2d_stage].reorder(m, l, n, xo, ko, ki, xi)
    s[avg_pool2d_stage].tensorize(ki, env.intrins.get('VAddMerge', mode = modes))
    s[cdata].compute_at(s[avg_pool2d_stage], ko)
    s[c2data].compute_at(s[avg_pool2d_stage], ko)
    ko, ki = s[avg_pool2d_stage1].split(s[avg_pool2d_stage1].op.reduce_axis[0], factor=1)
    xo, xi = s[avg_pool2d_stage1].split(s[avg_pool2d_stage1].op.axis[3], factor = factors)
    m, l, n = s[avg_pool2d_stage1].op.axis[0:3]
    s[avg_pool2d_stage1].reorder(m, l, n, xo, ko, ki, xi)
    s[avg_pool2d_stage1].tensorize(ki, env.intrins.get('VAddMerge', mode = modes, nDim = 3))

    s[avg_pool2d_stage].compute_at(s[avg_pool2d_stage1], ko)

    xo, xi = s[avg_pool2d_stage2].split(s[avg_pool2d_stage2].op.axis[3], factor = factors)
    m, l, n = s[avg_pool2d_stage2].op.axis[0:3]
    s[avg_pool2d_stage2].reorder(m, l, n, xo, xi)
    if modes == 'w':
        s[avg_pool2d_stage2].tensorize(xi, env.intrins.get('VDivI', imm_value = Imm.value, mode = modes))
    elif modes == 'n':
        s[avg_pool2d_stage2].tensorize(xi, env.intrins.get('VDivI', imm_value = Imm.value, mode = 'inc'))
    s[avg_pool2d_stage1].compute_at(s[avg_pool2d_stage2], xo)
    s[avg_pool2d_stage2].compute_at(s[cout], s[cout].op.axis[2])
    print(nnpu.lower(s, [data, output], simple_mode = True))
    return s
    
@reg.register_schedule("global_avg_pool2d", level = levels)
def schedule_global_avg_pool2d(attrs, outs, target):
    layout = attrs["layout"]
    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "nnpu":
            return schedule_global_avg_pool2d_default(attrs, outs)
        if str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        raise RuntimeError("not support target %s"%target)
    return _nn.schedule_global_avg_pool2d(attrs, outs, target)

def compute_flatten_default(data):
    ishape = data.shape
    dim = 1
    for i in range(1, len(data.shape)):
        dim = dim * ishape[i]
    oshape = [ishape[0], dim]
    def unwrap(idx, shape):
        index = []
        for s in reversed(shape):
            index.append(idx % s)
            idx = idx / s
        return list(reversed(index))
    return tvm.compute(oshape, lambda i, j : data(i, *unwrap(j, ishape[1: ])))

def schedule_flatten_default(outs):
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    flatten_res = []
    flatten_res.append(output.op)
    assert len(flatten_res) == 1
    flatten_stage = flatten_res[0].output(0)
    data = flatten_stage.op.input_tensors[0]
    data_shape = topi.util.get_const_tuple(data.shape)
    if data.dtype == dtype_n:
        modes = 'n'
    elif data.dtype == dtype_w:
        modes = 'w'
    s = nnpu.create_schedule(output.op)
    cout = s.cache_write(output, env.dram_scope)
    flatten_stage = s.cache_write(cout, env.uni_scratchpad_scope)
    
    cdata = s.cache_read(data, env.dram_scope, [flatten_stage])
    c2data = s.cache_read(cdata, env.uni_scratchpad_scope, [flatten_stage])
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy_pragma)
    s[c2data].pragma(s[c2data].op.axis[0], env.scratchpad_ls)

    s[flatten_stage].set_scope(env.uni_scratchpad_scope)
    
    s[cout].pragma(s[cout].op.axis[0], env.scratchpad_ls)
    s[output].pragma(s[output].op.axis[0], env.dma_copy_pragma)
    ko, k1 = s[flatten_stage].split(s[flatten_stage].op.axis[1], factor = data_shape[3])
    k3, k2 = s[flatten_stage].split(ko, factor = data_shape[2])
    s[flatten_stage].pragma(s[flatten_stage].op.axis[0], env.scratchpad_copy)
    return s

@reg.register_compute("flatten", level = levels)
def compute_flatten(attrs, inputs, out):
    return compute_flatten_default(inputs[0])

@reg.register_schedule("flatten", level = levels)
def schedule_flatten(attrs, outs, target):
    target = tvm.target.create(target)
    if target.device_name == "nnpu":
        return schedule_flatten_default(outs)
    if str(target).startswith("llvm"):
        return tvm.create_schedule([x.op for x in outs])
    raise RuntimeError("not support target %s"%target)