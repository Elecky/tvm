import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='test of NNPU Op')
parser.add_argument('--sim', type=str, help='the simulator to use', 
                    default='S0', choices=['S0', 'S1', 'SC'])
args = parser.parse_args()

env = nnpu.get_env()
nnpu.set_device(env, type=args.sim)

with ScheduleProcHelper():
    env = nnpu.get_env()
    shape1 = (8, 32)  # (8, 32) reshaped & transpoased to (4, 8, 8)
    shape2 = (16, 32)
    gemm_shape = (8, 8, 16)
    factor = gemm_shape[1]
    assert shape1[1] == shape2[1], \
        'gemm do dot product between rows, so the shape[1] of inputs should match'
    assert shape1[0] % gemm_shape[0] == 0, 'gemm insn require size of input 1 be x{0}'.format(gemm_shape[0])
    assert shape2[0] % gemm_shape[2] == 0, 'gemm insn require size of input 2 be x{0}'.format(gemm_shape[0])
    assert shape1[1] % factor == 0, 'gemm insn requires size of reduce dim be multiples of {0}'.format(factor)
    assert shape1[0] * shape2[0] % env.cfg['vector_unit']['size'] == 0, 'aha~~'
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    shape1_tiled = (shape1[1] // factor, shape1[0], factor)
    shape2_tiled = (shape2[1] // factor, shape2[0], factor)
    res_shape = (shape1[0], shape2[0])

    a = tvm.placeholder(shape1_tiled, dtype_n, 'a')
    b = tvm.placeholder(shape2_tiled, dtype_n, 'b')
    bias = tvm.placeholder((shape2[0], ), dtype_w, 'bias')

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a')
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b')
    # bias_buf, _ = nnpu.utils.CopyHtoBuf(bias, 'bias')

    ko = tvm.reduce_axis((0, shape1_tiled[0]), 'k0')
    ki = tvm.reduce_axis((0, factor), 'k0')

    res_acc = tvm.compute(res_shape, 
                            lambda i, j: tvm.sum(
                                a_buf[ko, i, ki].astype(dtype_w) * b_buf[ko, j, ki].astype(dtype_w),
                                axis=[ko, ki]))
    nnpu.utils.MarkScope(res_acc, 'acc')

    res_buf = nnpu.utils.CopyAccToBuf(res_acc, 'res')
    res_host, _ = nnpu.utils.CopyBufToH(res_buf, 'res')

    # res = tvm.compute(res_shape, 
    #                   lambda i, j: res_buf[i, j] + bias_buf[j],
    #                   'out')
    # nnpu.utils.MarkScope(res, 'buffer0')

    # res_host, _ = nnpu.utils.CopyBufToH(res, 'res')

    s = nnpu.create_schedule(res_host.op)
    print(tvm.lower(s, [a, b, res_host], simple_mode=True))
    
    # tensorize
    xi, xj = s[res_acc].op.axis
    ko, ki = s[res_acc].op.reduce_axis
    s[res_acc].reorder(ko, xi, xj, ki)
    s[res_acc].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, mode='inc', scope_out='acc'))

    # oco, oci = s[res].split(res.op.axis[1], 16)
    # s[res].tensorize(oci, env.intrins.get("VAddV", mode='w'))

    print(nnpu.lower(s, [a, b, res_host], simple_mode=True))
    # exit()
    func = nnpu.build(s, [a, b, res_host], 'nnpu', 'llvm', 'nnpu_func')
    print('------------------- device module 1 TVM IR: ')
    print(func.imported_modules[0].get_source('ir'))
    print('------------------- device module 1 uop: ')
    print(func.imported_modules[0].get_source('uop'))
    exit()

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape1_tiled, dtype=a.dtype, low = -32, high = 32)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=shape2_tiled, dtype=b.dtype, low = -32, high = 32)
    b_nd = tvm.nd.array(b_np, ctx)

    out_nd = tvm.nd.array(np.zeros(res_shape, dtype=res_host.dtype), ctx)

    func(a_nd, b_nd, out_nd)
    #print(out_nd.asnumpy())
    a_np = np.transpose(a_np, axes=(1, 0, 2))
    a_np = np.reshape(a_np, newshape=(shape1))
    b_np = np.transpose(b_np, axes=(1, 0, 2))
    b_np = np.reshape(b_np, newshape=(shape2))
    gt = np.dot(a_np, np.transpose(b_np, axes=(1, 0)).astype(dtype_w))
    np.testing.assert_allclose(out_nd.asnumpy(), gt)
    print('test passed')