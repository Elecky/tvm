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
    shape1 = (8, 32)  # (32, 32) reshaped to (32, 2, 16)
    shape2 = (8, 32)
    gemm_shape = (8, 16, 8)
    factor = gemm_shape[1]
    assert shape1[1] == shape2[1], \
           'gemm do dot product between rows, so the shape[1] of inputs should match'
    assert shape1[0] % gemm_shape[0] == 0, 'gemm insn require size of input 1 be x{0}'.format(gemm_shape[0])
    assert shape2[0] % gemm_shape[2] == 0, 'gemm insn require size of input 2 be x{0}'.format(gemm_shape[0])
    assert shape1[1] % factor == 0, 'gemm insn requires size of reduce dim be multiples of {0}'.format(factor)
    assert shape1[0] * shape2[0] % env.cfg['vector_unit']['size'] == 0, 'aha~~'
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder(shape1, dtype_n, 'a')
    b = tvm.placeholder(shape2, dtype_n, 'b')

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a')
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b')

    k = tvm.reduce_axis((0, factor), 'k0')

    prod_shape = (shape1[1] / factor, shape1[0], shape2[0])
    k = tvm.reduce_axis((0, factor), 'k0')
    prod_buf = tvm.compute(prod_shape, 
                        lambda j, i, l: tvm.sum(a_buf[i, j * factor + k].astype(dtype_w) * 
                                            b_buf[l, j * factor + k].astype(dtype_w), k), 'prod')
    nnpu.utils.MarkScope(prod_buf)

    re_shape = (prod_shape[0], prod_shape[1] * prod_shape[2])
    print('reshape = {0}'.format(re_shape))
    rp_buf = nnpu.utils.reshape(prod_buf, re_shape)


    k = tvm.reduce_axis((0, re_shape[0]), 'k1')
    out_buf = tvm.compute((re_shape[1], ), lambda i: tvm.sum(rp_buf[k, i], axis=k), 'out')
    nnpu.utils.MarkScope(out_buf)

    res = nnpu.utils.reshape(out_buf, (prod_shape[1], prod_shape[2]))
    res_host, _ = nnpu.utils.CopyBufToH(res, 'res')

    s = nnpu.create_schedule(res_host.op)

    # THIS MAY BE A HARD PROBLEM!!! inject_copy_intrin JUST CAN'T MERGE AXES AUTOMATICALLY
    # IF WE USE topi.reshape AS UNDERLYING IMPLEMENTION!!!
    s[rp_buf].split(rp_buf.op.axis[1], prod_shape[2])
    
    # tensorize
    xo, xi = s[prod_buf].split(prod_buf.op.axis[1], factor=gemm_shape[0])
    yo, yi = s[prod_buf].split(prod_buf.op.axis[2], factor=gemm_shape[2])
    s[prod_buf].reorder(xo, yo, prod_buf.op.axis[0], xi, yi, prod_buf.op.reduce_axis[0])
    s[prod_buf].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, mode='inc'))

    xo, xi = s[out_buf].split(out_buf.op.axis[0], env.cfg['vector_unit']['size'])
    ro, ri = s[out_buf].split(out_buf.op.reduce_axis[0], factor=1)
    s[out_buf].reorder(xo, ro, ri, xi)
    s[out_buf].tensorize(ri, env.intrins.get('VAddMerge', mode='w'))

    print(nnpu.lower(s, [a, b, res_host], simple_mode=True))

    func = nnpu.build(s, [a, b, res_host], 'nnpu', 'llvm', 'nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape1, dtype=a.dtype, low = -32, high = 32)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=shape2, dtype=b.dtype, low = -32, high = 32)
    b_nd = tvm.nd.array(b_np, ctx)

    out_nd = tvm.nd.array(np.zeros((shape1[0], shape2[0]), dtype=res_host.dtype), ctx)

    func(a_nd, b_nd, out_nd)
    #print(out_nd.asnumpy())
    gt = np.dot(a_np, np.transpose(b_np, axes=(1, 0)).astype(dtype_w))
    np.testing.assert_allclose(out_nd.asnumpy(), gt)
    print('test passed')