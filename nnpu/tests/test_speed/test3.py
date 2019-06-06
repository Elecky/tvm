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
    shape1 = (128, 256)
    shape2 = (128, 256)
    gemm_shape = (4, 4, 4)
    # gemm_shape = (8, 8, 8)
    factor = gemm_shape[1]
    assert shape1[1] == shape2[1], \
        'gemm do dot product between rows, so the shape[1] of inputs should match'
    assert shape1[0] % gemm_shape[0] == 0, 'gemm insn require size of input 1 be x{0}'.format(gemm_shape[0])
    assert shape2[0] % gemm_shape[2] == 0, 'gemm insn require size of input 2 be x{0}'.format(gemm_shape[0])
    assert shape1[1] % factor == 0, 'gemm insn requires size of reduce dim be multiples of {0}'.format(factor)

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    shape1_tiled = (shape1[0] // gemm_shape[0], shape1[1] // factor, 
                    gemm_shape[0], factor)
    shape2_tiled = (shape2[0] // gemm_shape[2], shape2[1] // factor,
                    gemm_shape[2], factor)
    
    a = tvm.placeholder(shape1_tiled, dtype_n, 'a')
    b = tvm.placeholder(shape2_tiled, dtype_n, 'b')

    a_buffer_scope = 'buffer1'
    b_buffer_scope = 'buffer0'

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', dst_scope=a_buffer_scope)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', dst_scope=b_buffer_scope)

    out_shape_tiled = (shape1_tiled[0], shape2_tiled[0], shape1_tiled[2], shape2_tiled[2])
    ko = tvm.reduce_axis((0, shape1[1] // factor), 'ko')
    ki = tvm.reduce_axis((0, factor), 'ki')

    out_acc = tvm.compute(out_shape_tiled, 
                          lambda xo, yo, xi, yi:
                            tvm.sum(a_buf[xo, ko, xi, ki].astype(dtype_w) 
                                    * b_buf[yo, ko, yi, ki].astype(dtype_w),
                                    axis=[ko, ki]),
                          'out')
    nnpu.utils.MarkScope(out_acc, 'acc')
    out_buf = tvm.compute(out_shape_tiled, lambda *i: out_acc(*i), 'out_host')
    nnpu.utils.MarkScope(out_buf)
    out_host = tvm.compute(out_shape_tiled, lambda *i: out_buf(*i), 'out_host')

    s = nnpu.create_schedule(out_host.op)

    # tensorize
    xo, yo, xi, yi = out_acc.op.axis
    ko, ki = out_acc.op.reduce_axis
    s[out_acc].reorder(xo, yo, ko, xi, yi, ki)
    s[out_acc].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, mode='inc', 
                                            scope_out='acc', scope_in1=a_buffer_scope,
                                            scope_in2=b_buffer_scope))

    s[out_buf].pragma(out_buf.op.axis[2], env.copy_acc2buf)

    # split output
    xo, yo, xi, yi = out_host.op.axis
    xparts, yparts = 1, 32
    xoo, xoi = s[out_host].split(xo, nparts=xparts)
    yoo, yoi = s[out_host].split(yo, factor=1)
    s[out_host].reorder(xoo, yoo, xoi, yoi, xi, yi)
    s[out_host].pragma(xi, env.dma_copy_from_buf)

    # compute_at
    s[a_buf].compute_at(s[out_host], xoo)
    s[b_buf].compute_at(s[out_host], yoo)

    s[out_buf].compute_at(s[out_host], yoi)
    s[out_acc].compute_at(s[out_host], yoi)
    # s[out_acc].unroll(s[out_acc].leaf_iter_vars[2])

    print(nnpu.lower(s, [a, b, out_host], simple_mode=True))

    func = nnpu.build(s, [a, b, out_host], 'nnpu', 'llvm', 'nnpu_func')
    # print('------------------- device module 1 asm code: ')
    # print(func.imported_modules[0].get_source('asm'))

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape1_tiled, dtype=a.dtype, low = -16, high = 16)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=shape2_tiled, dtype=b.dtype, low = -16, high = 16)
    b_nd = tvm.nd.array(b_np, ctx)

    out_nd = tvm.nd.array(np.zeros(out_shape_tiled, dtype=out_host.dtype), ctx)

    func(a_nd, b_nd, out_nd)