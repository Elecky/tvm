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
    shape1 = (128, 1024)
    shape2 = (128, 1024)
    gemm_shape = (8, 8, 8)
    factor = gemm_shape[1]
    assert shape1[1] == shape2[1], \
        'gemm do dot product between rows, so the shape[1] of inputs should match'
    assert shape1[0] % gemm_shape[0] == 0, 'gemm insn require size of input 1 be x{0}'.format(gemm_shape[0])
    assert shape2[0] % gemm_shape[2] == 0, 'gemm insn require size of input 2 be x{0}'.format(gemm_shape[0])
    assert shape1[1] % factor == 0, 'gemm insn requires size of reduce dim be multiples of {0}'.format(factor)

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    
    a = tvm.placeholder(shape1, dtype_n, 'a')
    b = tvm.placeholder(shape2, dtype_n, 'b')

    k = tvm.reduce_axis((0, shape1[1]), 'k')
    out_shape = (shape1[0], shape2[0])
    out_acc = tvm.compute(out_shape, 
                          lambda x, y:
                            tvm.sum(a[x, k].astype(dtype_w) * b[y, k].astype(dtype_w),
                                    axis=[k]),
                          'out_acc')

    out_buf = tvm.compute(out_shape, lambda *i: out_acc(*i), 'out_host')
    # nnpu.utils.MarkScope(out_buf)
    out_host = tvm.compute(out_shape, lambda *i: out_buf(*i), 'out_host')

    # schedule
    s = nnpu.create_schedule(out_host.op)
    al = s.cache_read(a, env.get_scope('buffer1'), out_acc)
    bl = s.cache_read(b, env.get_scope('buffer2'), out_acc)

    a_buffer_scope = 'buffer1'
    b_buffer_scope = 'buffer2'

    # set scope
    s[out_acc].set_scope(env.get_scope('acc'))
    s[out_buf].set_scope(env.get_scope('buffer3'))

    # pragma read
    s[al].pragma(al.op.axis[0], env.dma_copy_to_buf)
    s[bl].pragma(bl.op.axis[0], env.dma_copy_to_buf)

    # tensorize
    x, y = out_acc.op.axis
    xo, yo, xi, yi = s[out_acc].tile(x, y, gemm_shape[0], gemm_shape[2])
    ko, ki = s[out_acc].split(out_acc.op.reduce_axis[0], factor)
    koo, koi = s[out_acc].split(ko, 4)
    s[out_acc].reorder(koo, xo, yo, koi, xi, yi, ki)
    s[out_acc].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, mode='inc', 
                                            scope_out='acc', scope_in1='buffer1',
                                            scope_in2='buffer2'))
    s[al].compute_at(s[out_acc], koo)
    s[bl].compute_at(s[out_acc], koo)

    s[out_buf].pragma(out_buf.op.axis[0], env.copy_acc2buf)

    # split output
    x, y = out_host.op.axis
    dim_x, dim_y = 128, 128  # this the the rows of matrix loaded to faster scratchpad
    xo, yo, xi, yi = s[out_host].tile(x, y, dim_x, dim_y)
    s[out_host].reorder(xo, yo, xi, yi)
    s[out_host].pragma(xi, env.dma_copy_from_buf)

    # bind to virtual thread
    # s[out_host].bind(bx, tvm.thread_axis("cthread"))

    # compute_at

    s[out_buf].compute_at(s[out_host], yo)
    s[out_acc].compute_at(s[out_host], yo)

    # print(tvm.lower(s, [a, b, out_host], simple_mode=True))
    print(nnpu.lower(s, [a, b, out_host], simple_mode=True))

    func = nnpu.build(s, [a, b, out_host], 'nnpu', 'llvm', 'nnpu_func')
    print('------------------- device module 1 TVM IR: ')
    print(func.imported_modules[0].get_source('ir'))
    print('------------------- device module 1 uop: ')
    print(func.imported_modules[0].get_source('uop'))

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape1, dtype=a.dtype, low = -16, high = 16)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=shape2, dtype=b.dtype, low = -16, high = 16)
    b_nd = tvm.nd.array(b_np, ctx)

    out_nd = tvm.nd.array(np.zeros(out_shape, dtype=out_host.dtype), ctx)

    func(a_nd, b_nd, out_nd)