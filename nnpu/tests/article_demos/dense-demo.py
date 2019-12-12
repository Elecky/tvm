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
args = parser.parse_args()

if (args.profile):
    profile_dir = '/home/jian/Documents/nnpu_profile'
    nnpu.set_profile(['timeline', 'memory_access_latency'], profile_dir)

def roundup(x, d):
    return (x + d - 1) // d * d

with ScheduleProcHelper(), nnpu.Environment('./nnpu_config-dense.yaml'):
    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)

    shape1 = (128, 512)
    shape2 = (129, 512)

    macops = shape1[0] * shape1[1] * shape2[0]

    # upround shapes
    gemm_shape = (8, 8, 8)
    # shape1 = (roundup(shape1[0], gemm_shape[0]), shape1[1])
    shape2 = (roundup(shape2[0], gemm_shape[2]), shape2[1])

    bias_shape = (shape2[0], )
    factor = gemm_shape[1]
    assert shape1[1] == shape2[1], \
        'gemm do dot product between rows, so the shape[1] of inputs should match'
    assert shape1[0] % gemm_shape[0] == 0, 'gemm insn require size of input 1 be x{0}'.format(gemm_shape[0])
    assert shape2[0] % gemm_shape[2] == 0, 'gemm insn require size of input 2 be x{0}'.format(gemm_shape[0])
    assert shape1[1] % factor == 0, 'gemm insn requires size of reduce dim be multiples of {0}'.format(factor)

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    
    a = tvm.placeholder(shape1, dtype_n, 'a')
    b = tvm.placeholder(shape2, dtype_n, 'b')
    bias = tvm.placeholder(bias_shape, dtype_w, 'bias')

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
                          'out_buf')
    out_acc = out_buf
    res_buf = tvm.compute(out_shape_tiled, lambda no, oco, ni, oci: (out_buf[no, oco, ni, oci] + bias[oco * factor + oci]).astype(dtype_n))
    # nnpu.utils.MarkScope(out_acc, 'acc')
    # out_buf = tvm.compute(out_shape_tiled, lambda *i: out_acc(*i), 'out_host')
    # nnpu.utils.MarkScope(out_buf)
    out_host = tvm.compute(out_shape_tiled, lambda *i: res_buf(*i), 'out_host')

    # schedule
    s = nnpu.create_schedule(out_host.op)
    # al = s.cache_read(a_buf, env.get_scope('buffer1'), out_acc)
    # bl = s.cache_read(b_buf, env.get_scope('buffer2'), out_acc)
    al = a_buf
    bl = b_buf
    bias_buf = s.cache_read(bias, env.get_scope('buffer5'), res_buf)

    a_buffer_scope = 'buffer1'
    b_buffer_scope = 'buffer2'

    # set scope
    s[a_buf].set_scope(env.get_scope(a_buffer_scope))
    s[b_buf].set_scope(env.get_scope(b_buffer_scope))
    s[out_buf].set_scope(env.get_scope('buffer3'))
    s[res_buf].set_scope(env.get_scope('buffer4'))

    # pragma read
    s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy_to_buf)
    s[b_buf].pragma(b_buf.op.axis[0], env.dma_copy_to_buf)
    s[bias_buf].pragma(bias_buf.op.axis[0], env.dma_copy_to_buf)
    # pragma copy
    # s[al].pragma(al.op.axis[0], env.scratchpad_copy)
    # s[bl].pragma(bl.op.axis[0], env.scratchpad_copy)

    # tensorize
    xo, yo, xi, yi = out_acc.op.axis
    ko, ki = out_acc.op.reduce_axis
    koo, koi = s[out_acc].split(ko, 4)
    s[out_acc].reorder(koo, xo, yo, koi, xi, yi, ki)
    s[out_acc].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, mode='inc', 
                                            scope_out='buffer3', scope_in1='buffer1',
                                            scope_in2='buffer2'))
    s[al].compute_at(s[out_acc], koo)
    s[bl].compute_at(s[out_acc], koo)

    s[res_buf].pragma(res_buf.op.axis[2], 'nnpu.vector', str({'code': 'matrix-vector', 'shape': (gemm_shape[0], gemm_shape[2])}))
    
    # split output
    xo, yo, tx, ty = out_host.op.axis
    # xparts, yparts = 2, 16
    # xo, xi = s[out_host].split(xo, nparts=xparts)
    # yo, yi = s[out_host].split(yo, nparts=yparts)
    dim_x, dim_y = 8, 16  # this the the rows of matrix loaded to faster scratchpad
    xo, yo, xi, yi = s[out_host].tile(xo, yo, dim_x, dim_y)
    factor_x, factor_y = 1, 1  # to split outter loop
    bx, by, xo, yo = s[out_host].tile(xo, yo, factor_x, factor_y)
    s[out_host].reorder(bx, by, yo, xo, xi, yi, tx, ty)
    s[out_host].pragma(xi, env.dma_copy_from_buf)

    # bind to virtual thread
    s[out_host].bind(bx, tvm.thread_axis("cthread"))
    s[bias_buf].compute_at(s[out_host], bx)

    # compute_at
    # s[a_buf].compute_at(s[out_host], bx)
    # s[b_buf].compute_at(s[out_host], yo)

    s[out_buf].compute_at(s[out_host], xo)
    s[res_buf].compute_at(s[out_host], xo)
    # s[out_acc].compute_at(s[out_host], xo)

    print(nnpu.lower(s, [a, b, bias, out_host], simple_mode=True))

    func = nnpu.build(s, [a, b, bias, out_host], 'nnpu', 'llvm', 'nnpu_func')
    print('------------------- device module 1 TVM IR: ')
    print(func.imported_modules[0].get_source('ir'))
    print('------------------- device module 1 uop: ')
    print(func.imported_modules[0].get_source('uop'))

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape1, dtype=a.dtype, low = -16, high = 16)
    # a_np = np.ones(shape1, dtype=a.dtype)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=shape2, dtype=b.dtype, low = -16, high = 16)
    # b_np = np.ones(shape2, dtype=b.dtype)
    b_nd = tvm.nd.array(b_np, ctx)
    bias_np = np.random.randint(size=(shape2[0], ), dtype=bias.dtype, low = -128, high = 127)
    # bias_np = np.zeros((shape2[1], ), dtype=bias.dtype)
    bias_nd= tvm.nd.array(bias_np, ctx)

    out_nd = tvm.nd.array(np.zeros(out_shape_tiled, dtype=out_host.dtype), ctx)

    func(a_nd, b_nd, bias_nd, out_nd)

    gt = np.matmul(a_np, b_np.transpose(), dtype='int16')
    gt = gt + bias_np
    gt = gt.astype(np.int8)
    out_np = out_nd.asnumpy()
    # print(out_np)
    out_np = np.transpose(out_np, axes=(0, 2, 1, 3))
    out_np = np.reshape(out_np, (shape1[0], shape2[0]))
    np.testing.assert_allclose(gt, out_np)
    print('test passed')

    from functools import reduce 
    if (args.sim == 'SC'):
        print()
        print('###### summary ######')
        print('total mac ops =', macops)
        print('ellapsed cycles =', nnpu.get_cycle())
        print('efficiency =', macops / nnpu.get_cycle() / reduce(lambda x, y: x*y, gemm_shape, 1))
        if (args.profile):
            print('timeline saved in', profile_dir)