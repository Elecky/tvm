import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='test gemm with tiled/non-tiled data')
parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
args = parser.parse_args()

with ScheduleProcHelper(), nnpu.Environment('./nnpu_config_fp32.yaml'):
    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    assert dtype_w in ['float32', 'float16'], 'when testing activation function, float dtype is needed'

    shape = (64, )
    a = tvm.placeholder(shape, dtype_w, 'a')
    a_buf = tvm.compute(shape, lambda *i: a(*i), 'a_buf')

    exp = tvm.compute(shape, lambda i: tvm.exp(a_buf[i]), 'exp')
    log = tvm.compute(shape, lambda i: tvm.log(a_buf[i]), 'exp')
    tanh = tvm.compute(shape, lambda i: tvm.tanh(a_buf[i]), 'exp')
    sigmoid = tvm.compute(shape, lambda i: tvm.sigmoid(a_buf[i]), 'exp')

    # k = tvm.reduce_axis((0, 16), 'k0')
    # sum = tvm.compute((1, ), lambda i: tvm.sum(sigmoid[k], axis=k), 'sum')
    # nnpu.utils.MarkScope(sum)

    # softmax = tvm.compute(shape, lambda i: sigmoid[i] / sum[0], 'softmax')
    # nnpu.utils.MarkScope(softmax)
    # softmax_host, _ = nnpu.utils.CopyBufToH(softmax, 'softmax')

    s = nnpu.create_schedule([exp.op, log.op, tanh.op, sigmoid.op])
    # cache write
    exp_buf = s.cache_write(exp, env.get_scope('buffer0'))
    log_buf = s.cache_write(log, env.get_scope('buffer0'))
    tanh_buf = s.cache_write(tanh, env.get_scope('buffer0'))
    sigmoid_buf = s.cache_write(sigmoid, env.get_scope('buffer0'))
    # set scope
    s[a_buf].set_scope(env.get_scope('buffer0'))
    # pragma
    s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy_to_buf)
    s[exp].pragma(exp.op.axis[0], env.dma_copy_from_buf)
    s[log].pragma(log.op.axis[0], env.dma_copy_from_buf)
    s[tanh].pragma(tanh.op.axis[0], env.dma_copy_from_buf)
    s[sigmoid].pragma(sigmoid.op.axis[0], env.dma_copy_from_buf)
    # tensorize
    vector_unit_size = 32
    xo, xi = s[exp_buf].split(exp_buf.op.axis[0], vector_unit_size)
    s[exp_buf].tensorize(xi, env.intrins.get('VExp', mode='w', size=vector_unit_size))

    xo, xi = s[log_buf].split(log_buf.op.axis[0], vector_unit_size)
    s[log_buf].tensorize(xi, env.intrins.get('VLog', mode='w', size=vector_unit_size))

    xo, xi = s[tanh_buf].split(tanh_buf.op.axis[0], vector_unit_size)
    s[tanh_buf].tensorize(xi, env.intrins.get('VTanh', mode='w', size=vector_unit_size))

    xo, xi = s[sigmoid_buf].split(sigmoid_buf.op.axis[0], vector_unit_size)
    s[sigmoid_buf].tensorize(xi, env.intrins.get('VSigmoid', mode='w', size=vector_unit_size))

    print(nnpu.lower(s, [a, exp, log, tanh, sigmoid], simple_mode=True))

    func = nnpu.build(s, [a, exp, log, tanh, sigmoid], 'nnpu', 'llvm', 'nnpu_func')
    print('------------------- device module 1 IR: ')
    print(func.imported_modules[0].get_source('ir'))

    print('------------------- device module 1 micro code: ')
    print(func.imported_modules[0].get_source('uop'))
    # exit()

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.random(shape).astype(a.dtype) * 2
    a_nd = tvm.nd.array(a_np, ctx)

    exp_nd = tvm.nd.array(np.zeros(shape, dtype=exp.dtype), ctx)
    log_nd = tvm.nd.array(np.zeros(shape, dtype=log.dtype), ctx)
    tanh_nd = tvm.nd.array(np.zeros(shape, dtype=tanh.dtype), ctx)
    sigmoid_nd = tvm.nd.array(np.zeros(shape, dtype=sigmoid.dtype), ctx)

    func(a_nd, exp_nd, log_nd, tanh_nd, sigmoid_nd)

    if (dtype_n == 'float16'):
        rtol = 5e-2
    else:
        rtol = 1e-6

    np.testing.assert_allclose(exp_nd.asnumpy(), np.exp(a_np), rtol=rtol)
    np.testing.assert_allclose(log_nd.asnumpy(), np.log(a_np), rtol=rtol)
    np.testing.assert_allclose(tanh_nd.asnumpy(), np.tanh(a_np), rtol=rtol)
    np.testing.assert_allclose(sigmoid_nd.asnumpy(), np.exp(a_np)/(1+np.exp(a_np)), rtol=rtol)
    print('test passed')