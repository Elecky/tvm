import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='test of NNPU Op')
parser.add_argument('--sim', type=str, help='the simulator to use', 
                    default='S0', choices=['S0', 'S1', 'SC'])
parser.add_argument('--profile', type=bool, help='enable profiling', 
                    default=True)
parser.add_argument('--parallel', type=bool, help='enable parallel', 
                    default=False)
parser.add_argument('--dimx', type=int, help='tile size of x', 
                    default=2)
parser.add_argument('--dimy', type=int, help='tile size of y', 
                    default=2)
args = parser.parse_args()

cfg_path = './nnpu_config-opt.yaml'
if (args.profile):
    profile_dir = '/home/jian/Documents/nnpu_profile'
    nnpu.set_profile(['timeline', 'memory_access_latency'], profile_dir)

with ScheduleProcHelper(), nnpu.Environment(cfg_path):
    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    
    shape1 = (128, 256)
    shape2 = (128, 256)
    macops = shape1[0] * shape1[1] * shape2[0]
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

    shape1_tiled = (shape1[0] // gemm_shape[0], shape1[1] // factor, 
                    gemm_shape[0], factor)
    shape2_tiled = (shape2[0] // gemm_shape[2], shape2[1] // factor,
                    gemm_shape[2], factor)
    a_buf = tvm.compute(shape1_tiled, lambda no, ico, ni, ici: a[no * gemm_shape[0] + ni, ico * factor + ici], 'a_buf')
    b_buf = tvm.compute(shape2_tiled, lambda oco, ico, oci, ici: b[oco * gemm_shape[2] + oci, ico * factor + ici], 'b_buf')

    out_shape_tiled = (shape1_tiled[0], shape2_tiled[0], shape1_tiled[2], shape2_tiled[2])
    ko = tvm.reduce_axis((0, shape1[1] // factor), 'ko')
    ki = tvm.reduce_axis((0, factor), 'ki')

    out_buf = tvm.compute(out_shape_tiled, 
                          lambda xo, yo, xi, yi:
                            tvm.sum(a_buf[xo, ko, xi, ki].astype(dtype_w) 
                                    * b_buf[yo, ko, yi, ki].astype(dtype_w),
                                    axis=[ko, ki]),
                          'out_acc')
    # out_buf = tvm.compute(out_shape_tiled, lambda *i: out_acc(*i), 'out_host')
    out_host = tvm.compute(out_shape_tiled, lambda *i: out_buf(*i), 'out_host')

    # schedule
    s = nnpu.create_schedule(out_host.op)
    al = a_buf
    bl = b_buf

    a_buffer_scope = 'buffer1'
    b_buffer_scope = 'buffer2'

    # set scope
    s[a_buf].set_scope(env.get_scope(a_buffer_scope))
    s[b_buf].set_scope(env.get_scope(b_buffer_scope))
    s[out_buf].set_scope(env.get_scope('buffer3'))
    # s[out_acc].set_scope(env.get_scope('acc'))

    # pragma read
    s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy_to_buf)
    s[b_buf].pragma(b_buf.op.axis[0], env.dma_copy_to_buf)

    # tensorize
    xo, yo, xi, yi = out_buf.op.axis
    ko, ki = out_buf.op.reduce_axis
    koo, koi = s[out_buf].split(ko, 4)
    s[out_buf].reorder(koo, xo, yo, koi, xi, yi, ki)
    s[out_buf].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, mode='inc', 
                                            scope_out='buffer3', scope_in1='buffer1',
                                            scope_in2='buffer2'))
    s[al].compute_at(s[out_buf], koo)
    s[bl].compute_at(s[out_buf], koo)

    # s[out_buf].pragma(out_buf.op.axis[0], env.copy_acc2buf)

    # split output
    xo, yo, tx, ty = out_host.op.axis
    # this the the rows of matrix loaded to faster scratchpad
    dim_x, dim_y = args.dimx, args.dimy
    xo, yo, xi, yi = s[out_host].tile(xo, yo, dim_x, dim_y)
    if (args.parallel):
        nparts_x, nparts_y = 2, 1  # to split outter loop
    else:
        nparts_x, nparts_y = 1, 1  # to split outter loop
    # bx, by, xo, yo = s[out_host].tile(xo, yo, factor_x, factor_y)
    bx, xo = s[out_host].split(xo, nparts=nparts_x)
    by, yo = s[out_host].split(yo, nparts=nparts_y)
    s[out_host].reorder(bx, by, yo, xo, xi, yi, tx, ty)
    s[out_host].pragma(xi, env.dma_copy_from_buf)

    # bind to virtual thread
    s[out_host].bind(bx, tvm.thread_axis("cthread"))

    # compute_at
    s[out_buf].compute_at(s[out_host], xo)
    # s[out_acc].compute_at(s[out_host], xo)

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

    out_nd = tvm.nd.array(np.zeros(out_shape_tiled, dtype=out_host.dtype), ctx)

    func(a_nd, b_nd, out_nd)

    from functools import reduce 
    if (args.sim == 'SC'):
        print()
        print('###### summary ######')
        print('total mac ops =', macops)
        print('ellapsed cycles =', nnpu.get_cycle())
        print('efficiency =', macops / nnpu.get_cycle() / reduce(lambda x, y: x*y, gemm_shape, 1))
        if (args.profile):
            print('timeline saved in', profile_dir)