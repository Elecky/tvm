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

env = nnpu.get_env()
nnpu.set_device(env, type=args.sim)

with (ScheduleProcHelper()):
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    assert dtype_w in ['float32', 'float16'], 'when testing activation function, float dtype is needed'

    shape = (128, )
    a = tvm.placeholder(shape, dtype_w, 'a')
    a_buf, _ = nnpu.utils.CopyHtoBuf(a, 'a')

    exp = tvm.compute(shape, lambda i: tvm.exp(a_buf[i]), 'exp')
    nnpu.utils.MarkScope(exp)

    one = tvm.const(1, dtype_w)
    exp_p1 = tvm.compute(shape, lambda i: exp[i] + one, 'exp_p1')
    nnpu.utils.MarkScope(exp_p1)

    sigmoid = tvm.compute(shape, lambda i: exp[i] / exp_p1[i], 'sigmoid')
    nnpu.utils.MarkScope(sigmoid)
    sigmoid_host, _ = nnpu.utils.CopyBufToH(sigmoid, 'sigmoid')

    # k = tvm.reduce_axis((0, 16), 'k0')
    # sum = tvm.compute((1, ), lambda i: tvm.sum(sigmoid[k], axis=k), 'sum')
    # nnpu.utils.MarkScope(sum)

    # softmax = tvm.compute(shape, lambda i: sigmoid[i] / sum[0], 'softmax')
    # nnpu.utils.MarkScope(softmax)
    # softmax_host, _ = nnpu.utils.CopyBufToH(softmax, 'softmax')

    s = nnpu.create_schedule([sigmoid_host.op])
    # tensorize
    xo, xi = s[exp].split(exp.op.axis[0], 16)
    s[exp].tensorize(xi, env.intrins.get('VExp', mode='w'))
    xo, xi = s[exp_p1].split(exp_p1.op.axis[0], 16)
    s[exp_p1].tensorize(xi, env.intrins.get('VAddI', mode='w'))
    xo, xi = s[sigmoid].split(sigmoid.op.axis[0], 16)
    s[sigmoid].tensorize(xi, env.intrins.get('VDivV', mode='w'))
    # s[sum].tensorize(sum.op.axis[0], env.intrins.get('VReduceSum', mode='w'))
    # s[softmax].tensorize(softmax.op.axis[0], env.intrins.get('VDivS', mode='w'))

    print(nnpu.lower(s, [a, sigmoid_host], simple_mode=True))

    func = nnpu.build(s, [a, sigmoid_host], 'nnpu', 'llvm', 'nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.random(shape).astype(a.dtype) * 2
    a_nd = tvm.nd.array(a_np, ctx)

    sigmoid_nd = tvm.nd.array(np.zeros(shape, dtype=sigmoid_host.dtype), ctx)
    # softmax_nd = tvm.nd.array(np.zeros(shape, dtype=softmax_host.dtype), ctx)

    func(a_nd, sigmoid_nd)

    if (dtype_n == 'float16'):
        rtol = 5e-2
    else:
        rtol = 1e-6

    print('a = ')
    print(a_np)
    print('sigmoid = ')
    print(sigmoid_nd.asnumpy())
    gt = np.exp(a_np) / (1 + np.exp(a_np))
    np.testing.assert_allclose(sigmoid_nd.asnumpy(), gt, rtol=rtol)

    # print('softmax = ')
    # res = softmax_nd.asnumpy()
    # print(res)
    # gt = gt / np.sum(gt)
    # np.testing.assert_allclose(res, gt, rtol=rtol)
    print('test passed')