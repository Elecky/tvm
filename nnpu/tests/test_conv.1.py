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

    shape = (32, 18, 32)  # (h, w, c)
    kshape = (3, 3, 48, 32)  # (kh, kw, oc, c)
    assert shape[-1] == kshape[-1], 'feature map in-channel != kernel in-channel'
    assert shape[0] >= kshape[0] and shape[1] >= kshape[1], 'feature map smaller than kernel'
    gemm_shape = (8, 8, 8)
    n_row, factor, n_col = gemm_shape
    assert shape[-1] % factor == 0, 'in-channel not divisible to factor'
    assert kshape[-2] % n_col == 0, 'out-channel not divisible to gemm insn NColOut'
    nvctr_unit = env.cfg['vector_unit']['size']

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    feature = tvm.placeholder(shape, dtype_n, 'feature-map')
    kernel = tvm.placeholder(kshape, dtype_n, 'kernel')

    shape_tiled = (shape[0], shape[2] // factor, shape[1], factor)
    kshape_tiled = (kshape[0], kshape[1], kshape[2] // n_col, kshape[3] // factor, n_col, factor)

    feature_buf = tvm.compute(shape_tiled, 
                              lambda h, co, w, ci: feature[h, w, co * factor + ci],
                              'feature_buf')

    kernel_buf = tvm.compute(kshape_tiled,
                             lambda kh, kw, oco, co, oci, ci:
                                kernel[kh, kw, oco * n_col + oci, co * factor + ci],
                             'kernel_buf')

    conv_shape = (shape[0] - kshape[0] + 1, shape[1] - kshape[1] + 1, kshape[2])
    conv_shape_tiled = (conv_shape[0], conv_shape[1] // n_row, conv_shape[2] // n_col, n_row, n_col)
    kh_reduce = tvm.reduce_axis((0, kshape[0]), 'kh.i')
    kw_reduce = tvm.reduce_axis((0, kshape[1]), 'kw.i')
    co_reduce = tvm.reduce_axis((0, shape_tiled[1]), 'co.i')
    ci_reduce = tvm.reduce_axis((0, factor), 'ci.i')
    conv_acc = tvm.compute(conv_shape_tiled,
                           lambda h, wo, oco, wi, oci:
                                tvm.sum(feature_buf[h + kh_reduce, co_reduce, wo * n_row + wi + kw_reduce, ci_reduce].astype(dtype_w) \
                                        * kernel_buf[kh_reduce, kw_reduce, oco, co_reduce, oci, ci_reduce].astype(dtype_w), 
                                        axis=[kh_reduce, kw_reduce, co_reduce, ci_reduce]),
                           'conv_acc')

    conv = tvm.compute(conv_shape,
                       lambda h, w, oc:
                            conv_acc[h, w / n_row, oc / n_col, w % n_row, oc % n_col],
                        'conv_buf')

    conv_host = tvm.compute(conv_shape, lambda *i: conv(*i), 'conv_host')

    # pragma scope
    nnpu.utils.MarkScope(feature_buf, 'buffer0')
    nnpu.utils.MarkScope(kernel_buf, 'buffer1')
    nnpu.utils.MarkScope(conv_acc, 'acc')
    nnpu.utils.MarkScope(conv, 'buffer0')

    s = nnpu.create_schedule(conv_host.op)

    # reorder and split
    h, wo, oco, wi, oci = s[conv_acc].op.axis
    s[conv_acc].reorder(h, wo, oco, kh_reduce, kw_reduce, co_reduce, wi, oci, ci_reduce)
    # tensorize
    s[conv_acc].tensorize(wi, env.intrins.get('GEMM', shape=gemm_shape, mode='inc', scope_in2='buffer1',
                                              scope_out='acc'))

    # compute_at
    s[conv_acc].compute_at(s[conv], conv.op.axis[0])
    # unroll
    s[conv_acc].unroll(co_reduce)
    s[conv_acc].unroll(kw_reduce)
    s[conv_acc].unroll(kh_reduce)

    # split conv to eliminate divide and module operator.
    io, ii = s[conv].split(conv.op.axis[1], n_row)
    s[conv].split(conv.op.axis[2], n_col)
    res_host = conv_host

    # pragma
    s[feature_buf].pragma(feature_buf.op.axis[0], env.dma_copy_to_buf)
    s[kernel_buf].pragma(kernel_buf.op.axis[0], env.dma_copy_to_buf)
    s[conv].pragma(io, env.copy_acc2buf)
    s[conv_host].pragma(conv_host.op.axis[0], env.dma_copy_from_buf)

    print(nnpu.lower(s, [feature, kernel, res_host], simple_mode=True))

    func = nnpu.build(s, [feature, kernel, res_host], 'nnpu', 'llvm', 'nnpu_conv')
    # print('------------------- device module 1 asm code: ')
    # print(func.imported_modules[0].get_source('asm'))

    ctx = tvm.nd.TVMContext(13, 0)
    fm_np = np.random.randint(size=shape, dtype=feature.dtype, low = -16, high = 16)
    fm_nd = tvm.nd.array(fm_np, ctx)

    k_np = np.random.randint(size=kshape, dtype=kernel.dtype, low = -16, high = 16)
    k_nd = tvm.nd.array(k_np, ctx)

    res_nd = tvm.nd.array(np.zeros(conv_shape, dtype=res_host.dtype), ctx)

    nnpu.set_dump(False)

    func(fm_nd, k_nd, res_nd)
    print('execute finish')

    res_np = res_nd.asnumpy()

# calculate ground truth
feature = tvm.placeholder(shape, dtype_n, 'feature-map')
kernel = tvm.placeholder(kshape, dtype_n, 'kernel')
res_shape = conv_shape  # (x, y, oc)
rc = tvm.reduce_axis((0, shape[-1]), 'rc')
ry = tvm.reduce_axis((0, kshape[1]), 'ry')
rx = tvm.reduce_axis((0, kshape[0]), 'rx')

res = tvm.compute(res_shape, 
                lambda x, y, oc: 
                    tvm.sum(feature[x + rx, y + ry, rc].astype(dtype_w) * 
                            kernel[rx, ry, oc, rc].astype(dtype_w), 
                            axis=[rx, ry, rc]),
                'res')
s1 = tvm.create_schedule(res.op)
#print(tvm.lower(s1, [feature, kernel, res], simple_mode=True))
h_func = tvm.build(s1, [feature, kernel, res], 'llvm', 'llvm', 'host_conv')

fm_nd = tvm.nd.array(fm_np)
k_nd = tvm.nd.array(k_np)

gt_nd = tvm.nd.array(np.zeros(res_shape, dtype=dtype_w))
h_func(fm_nd, k_nd, gt_nd)

np.testing.assert_allclose(res_nd.asnumpy(), gt_nd.asnumpy())
print('test passed')