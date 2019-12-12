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

profile_dir = '/home/jian/Documents/nnpu_profile'
# nnpu.set_profile(['timeline', 'memory_access_latency'], profile_dir)

cfg_path = '/home/jian/nnpu_config2.yaml'

gemm_shape = (8, 8, 8)
dim_x, dim_y = 16, 16
factor_x, factor_y = 16, 16

with ScheduleProcHelper(), nnpu.Environment(cfg_path):
    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    shape1 = (128, 256)
    shape2 = (128, 256)

    factor = gemm_shape[1]
    assert shape1[1] == shape2[1], \
        'gemm do dot product between rows, so the shape[1] of inputs should match'
    assert shape1[0] % gemm_shape[0] == 0, 'gemm insn require size of input 1 be x{0}'.format(gemm_shape[0])
    assert shape2[0] % gemm_shape[2] == 0, 'gemm insn require size of input 2 be x{0}'.format(gemm_shape[0])
    assert shape1[1] % factor == 0, 'gemm insn requires size of reduce dim be multiples of {0}'.format(factor)

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    
    a = tvm.placeholder(shape1, dtype_n, 'a')
    b = tvm.placeholder(shape2, dtype_n, 'b')
    bias = tvm.placeholder((shape2[0],), dtype_n, 'bias')

    shape1_tiled = (shape1[0] // gemm_shape[0], shape1[1] // factor, 
                    gemm_shape[0], factor)
    shape2_tiled = (shape2[0] // gemm_shape[2], shape2[1] // factor,
                    gemm_shape[2], factor)
    a_buf = tvm.compute(shape1_tiled, lambda no, ico, ni, ici: a[no * gemm_shape[0] + ni, ico * factor + ici], 'a_buf')
    b_buf = tvm.compute(shape2_tiled, lambda oco, ico, oci, ici: b[oco * gemm_shape[2] + oci, ico * factor + ici], 'b_buf')

    out_shape_tiled = (shape1_tiled[0], shape2_tiled[0], shape1_tiled[2], shape2_tiled[2])
    ko = tvm.reduce_axis((0, shape1[1] // factor), 'ko')
    ki = tvm.reduce_axis((0, factor), 'ki')

    out_acc = tvm.compute(out_shape_tiled, 
                          lambda xo, yo, xi, yi:
                            tvm.sum(a_buf[xo, ko, xi, ki] * b_buf[yo, ko, yi, ki], axis=[ko, ki]),
                          'out_acc')
    out_buf = out_acc
    # out_buf = tvm.compute(out_shape_tiled, lambda *i: out_acc(*i).astype(dtype_n), 'out_buf')
    res_buf = tvm.compute(out_shape_tiled, lambda no, oco, ni, oci: out_buf[no, oco, ni, oci] + bias[oco * factor + oci])
    out_host = tvm.compute(out_shape_tiled, lambda *i: res_buf(*i), 'out_host')

    # schedule
    s = nnpu.create_schedule(out_host.op)
    al = a_buf
    bl = b_buf
    bias_buf = s.cache_read(bias, env.get_scope('buffer5'), res_buf)

    a_buffer_scope = 'buffer1'
    b_buffer_scope = 'buffer2'

    # set scope
    s[a_buf].set_scope(env.get_scope(a_buffer_scope))
    s[b_buf].set_scope(env.get_scope(b_buffer_scope))
    s[out_buf].set_scope(env.get_scope('buffer3'))
    # s[out_acc].set_scope(env.get_scope('acc'))
    # s[bias_buf].set_scope(env.get_scope('buffer5'))
    s[res_buf].set_scope(env.get_scope('buffer4'))

    # pragma read
    s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy_to_buf)
    s[b_buf].pragma(b_buf.op.axis[0], env.dma_copy_to_buf)
    s[bias_buf].pragma(bias_buf.op.axis[0], env.dma_copy_to_buf)

    # tensorize
    xo, yo, xi, yi = out_acc.op.axis
    ko, ki = out_acc.op.reduce_axis
    koo, koi = s[out_acc].split(ko, 4)
    s[out_acc].reorder(koo, xo, yo, koi, xi, yi, ki)
    s[out_acc].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, mode='n', 
                                            scope_out='buffer3', scope_in1='buffer1',
                                            scope_in2='buffer2'))
    s[al].compute_at(s[out_acc], koo)
    s[bl].compute_at(s[out_acc], koo)

    # s[out_buf].pragma(out_buf.op.axis[0], env.copy_acc2buf)

    # tensorize add bias
    s[res_buf].pragma(res_buf.op.axis[2], 'nnpu.vector', str({'code': 'matrix-vector', 'shape': [8, 8]}))
    # s[res_buf].tensorize(res_buf.op.axis[2], 
    #                      env.intrins.get('MAddV', shape=(gemm_shape[0], gemm_shape[2]),
    #                                      scope_in_mat='buffer3', scope_in_vctr='buffer5', scope_out='buffer4', mode='n'))

    # split output
    xo, yo, tx, ty = out_host.op.axis
    # this the the rows of matrix loaded to faster scratchpad
    # dim_x, dim_y = 16, 8
    # dim_x, dim_y = 8, 4
    xo, yo, xi, yi = s[out_host].tile(xo, yo, dim_x, dim_y)
    # factor_x, factor_y = 1, 8  # to split outter loop
    # factor_x, factor_y = 1, 8  # to split outter loop
    s[out_host].reorder(xo, yo, xi, yi, tx, ty)
    s[out_host].pragma(yi, env.dma_copy_from_buf)

    # bind to virtual thread
    # s[out_host].bind(by, tvm.thread_axis("cthread"))

    # compute_at
    s[out_acc].compute_at(s[out_host], xo)
    # s[out_buf].compute_at(s[out_host], xi)
    s[res_buf].compute_at(s[out_host], xi)
    # s[bias_buf].compute_at(s[out_host], by)

    print(tvm.lower(s, [a, b, bias, out_host], simple_mode=True))
    # exit()
    print(nnpu.lower(s, [a, b, bias, out_host], simple_mode=True))
    # exit(0)
    func = nnpu.build(s, [a, b, bias, out_host], 'nnpu', 'llvm', 'nnpu_func')
    print('------------------- device module 1 TVM IR: ')
    print(func.imported_modules[0].get_source('ir'))
    print('------------------- device module 1 uop: ')
    print(func.imported_modules[0].get_source('uop'))

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape1, dtype=a.dtype, low = -16, high = 16)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=shape2, dtype=b.dtype, low = -16, high = 16)
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
    out_np = np.reshape(out_np, (128, 128))
    np.testing.assert_allclose(gt, out_np)
    print('test passed')