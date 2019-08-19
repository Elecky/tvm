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
    N = 128
    nRow, factor = (8, 8)
    shape = (N,)
    a = tvm.placeholder(shape, dtype_w, 'a')
    a_buf, _ = nnpu.utils.CopyHtoBuf(a, 'a')

    exp = tvm.compute(shape, lambda i: tvm.exp(a_buf[i]), 'exp')
    nnpu.utils.MarkScope(exp)

    one = tvm.const(1, dtype_w)
    exp_p1 = tvm.compute(shape, lambda i: exp[i] + one, 'exp_p1')
    nnpu.utils.MarkScope(exp_p1)

    sigmoid = tvm.compute(shape, lambda i: exp[i] / exp_p1[i], 'sigmoid')
    nnpu.utils.MarkScope(sigmoid)
    
    reshaped = (N // (nRow * factor), nRow, factor)
    sigmoid_re = tvm.compute(reshaped, lambda i, j, k: sigmoid[i*nRow*factor + j*nRow + k], 'sigmoid_reshaped')

    ko = tvm.reduce_axis((0, reshaped[0]), 'ko')
    ki = tvm.reduce_axis((0, factor), 'ki')
    sum1 = tvm.compute((nRow, ), lambda i: tvm.sum(sigmoid_re[ko, i, ki], axis=[ko, ki]), 'sum1')
    nnpu.utils.MarkScope(sum1, 'acc')
    sum1_buf = nnpu.utils.CopyAccToBuf(sum1, 'sum1')
    # sum1_buf = tvm.compute((nRow, ), lambda *i: sum1(*i))

    k = tvm.reduce_axis((0, nRow), 'k')
    sum2 = tvm.compute((1, ), lambda i: tvm.sum(sum1_buf[k], axis=k), 'sum2')
    nnpu.utils.MarkScope(sum2, 'buffer0')

    softmax = tvm.compute(shape, lambda i: sigmoid[i] / sum2[0], 'softmax')
    nnpu.utils.MarkScope(softmax)
    softmax_host, _ = nnpu.utils.CopyBufToH(softmax, 'softmax')

    s = nnpu.create_schedule([softmax_host.op])

    s[sigmoid_re].set_scope(env.scratchpad_scope(0))
    s[sigmoid_re].pragma(sigmoid_re.op.axis[0], env.scratchpad_copy)

    # tensorize
    xo, xi = s[exp].split(exp.op.axis[0], 16)
    s[exp].tensorize(xi, env.intrins.get('VExp', mode='w'))
    xo, xi = s[exp_p1].split(exp_p1.op.axis[0], 16)
    s[exp_p1].tensorize(xi, env.intrins.get('VAddI', mode='w'))
    xo, xi = s[sigmoid].split(sigmoid.op.axis[0], 16)
    s[sigmoid].tensorize(xi, env.intrins.get('VDivV', mode='w'))

    xblock, xcol = sum1.op.reduce_axis
    xrow = sum1.op.axis[0]
    s[sum1].reorder(xblock, xrow, xcol)
    s[sum1].tensorize(xrow, env.intrins.get('MReduceSumRow', shape= (nRow, factor), scope_out='acc', mode='w'))

    s[sum2].tensorize(sum2.op.reduce_axis[0], env.intrins.get('VReduceSum', shape=(8,), mode='w'))

    xo, xi = s[softmax].split(softmax.op.axis[0], 16)
    s[softmax].tensorize(xi, env.intrins.get('VDivS', mode='w'))

    print(nnpu.lower(s, [a, softmax_host], simple_mode=True))
    func = nnpu.build(s, [a, softmax_host], 'nnpu', 'llvm', 'nnpu_func')
    print('------------------- device module 1 IR: ')
    print(func.imported_modules[0].get_source('ir'))

    print('------------------- device module 1 micro code: ')
    print(func.imported_modules[0].get_source('uop'))
    # exit()

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.random(shape).astype(a.dtype) * 2
    a_nd = tvm.nd.array(a_np, ctx)

    # sigmoid_nd = tvm.nd.array(np.zeros(shape, dtype=sigmoid_host.dtype), ctx)
    softmax_nd = tvm.nd.array(np.zeros(shape, dtype=softmax_host.dtype), ctx)

    func(a_nd, softmax_nd)

    if (dtype_n == 'float16'):
        rtol = 5e-2
    else:
        rtol = 1e-6

    print('a = ')
    print(a_np)
    print('softmax = ')
    res = softmax_nd.asnumpy()
    print(res)
    gt = np.exp(a_np) / (1 + np.exp(a_np))
    # np.testing.assert_allclose(sigmoid_nd.asnumpy(), gt, rtol=rtol)

    gt = gt / np.sum(gt)
    np.testing.assert_allclose(res, gt, rtol=rtol)
    print('test passed')