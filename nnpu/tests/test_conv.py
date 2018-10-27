import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

with ScheduleProcHelper():
    env = nnpu.get_env()
    nnpu.set_device(env)

    shape = (9, 9, 32)  # (h, w, c)
    kshape = (3, 3, 48, 32)  # (kh, kw, oc, c)
    assert shape[-1] == kshape[-1], 'feature map in-channel != kernel in-channel'
    assert shape[0] >= kshape[0] and shape[1] >= kshape[1], 'feature map smaller than kernel'
    gemm_shape = (16, 16, 1)
    factor = gemm_shape[1]
    assert shape[-1] % factor == 0, 'in-channel not divisible to factor'
    assert kshape[-2] % gemm_shape[0] == 0, 'out-channel not divisible to gemm insn NRowOut'
    nvctr_unit = env.cfg['vector_unit']['size']

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    feature = tvm.placeholder(shape, dtype_n, 'feature-map')
    kernel = tvm.placeholder(kshape, dtype_n, 'kernel')

    f_buf, _ = nnpu.utils.CopyHtoBuf(feature, 'feature')
    k_buf, _ = nnpu.utils.CopyHtoBuf(kernel, 'kernel')

    out_shape = (shape[0] - kshape[0] + 1, shape[1] - kshape[1] + 1)
    imd1_shape = (out_shape[0], out_shape[1], kshape[0], kshape[1], shape[-1] / factor, kshape[2])
    k = tvm.reduce_axis((0, factor), 'k0')
    imd1 = tvm.compute(imd1_shape, 
                    lambda x, y, i, j, p, oc:
                        tvm.sum(k_buf[i, j, oc, p * factor + k].astype(dtype_w) * 
                                f_buf[x + i, y + j, p * factor + k].astype(dtype_w)
                                , axis=k),
                    'imd1')
    nnpu.utils.MarkScope(imd1)
    # sum all gemm parts up
    imd2_shape = (out_shape[0], out_shape[1], kshape[0], kshape[1], kshape[2])
    k = tvm.reduce_axis((0, shape[-1] / factor), 'k1')
    imd2 = tvm.compute(imd2_shape,
                    lambda x, y, i, j, oc:
                        tvm.sum(imd1[x, y, i, j, k, oc], axis = k),
                    'imd2')
    nnpu.utils.MarkScope(imd2)

    # sum 
    imd3_shape = (out_shape[0], out_shape[1], kshape[0], kshape[2])
    k = tvm.reduce_axis((0, kshape[1]), 'k2')
    imd3 = tvm.compute(imd3_shape,
                    lambda x, y, i, oc:
                        tvm.sum(imd2[x, y, i, k, oc], axis=k),
                    'imd3')
    nnpu.utils.MarkScope(imd3)

    imd4_shape = (out_shape[0], out_shape[1], kshape[2])  # (h, w, oc)
    k = tvm.reduce_axis((0, kshape[0]), 'k3')
    imd4 = tvm.compute(imd4_shape,
                    lambda x, y, oc:
                        tvm.sum(imd3[x, y, k, oc], axis=k),
                    'imd4')
    nnpu.utils.MarkScope(imd4)

    res_host, _ = nnpu.CopyBufToH(imd4, 'res')
    s = nnpu.create_schedule(res_host.op)
    # tensorize
    oco, oci = s[imd1].split(imd1.op.axis[5], factor = gemm_shape[0])
    s[imd1].tensorize(oci, env.intrins.get('GEMM', shape=gemm_shape, mode='inc', reduce=True))
    
    oco, oci = s[imd2].split(imd2.op.axis[4], factor=nvctr_unit)
    ko, ki = s[imd2].split(imd2.op.reduce_axis[0], factor=1)
    x, y, kx, ky = imd2.op.axis[0:4]
    s[imd2].reorder(x, y, kx, ky, oco, ko, ki, oci)
    s[imd2].tensorize(ki, env.intrins.get('VAddMerge', mode='w'))
    # schedule
    s[imd1].compute_at(s[imd2], ko)

    oco, oci = s[imd3].split(imd3.op.axis[3], factor=nvctr_unit)
    ko, ki = s[imd3].split(imd3.op.reduce_axis[0], factor=1)
    x, y, kx = imd3.op.axis[0:3]
    s[imd3].reorder(x, y, kx, oco, ko, ki, oci)
    s[imd3].tensorize(ki, env.intrins.get('VAddMerge', mode='w'))

    oco, oci = s[imd4].split(imd4.op.axis[2], factor=nvctr_unit)
    ko, ki = s[imd4].split(imd4.op.reduce_axis[0], factor=1)
    x, y = imd4.op.axis[0:2]
    s[imd4].reorder(x, y, oco, ko, ki, oci)
    s[imd4].tensorize(ki, env.intrins.get('VAddMerge', mode='w'))


    print(nnpu.lower(s, [feature, kernel, res_host], simple_mode=True))

    func = nnpu.build(s, [feature, kernel, res_host], 'nnpu', 'llvm', 'nnpu_conv')

    ctx = tvm.nd.TVMContext(13, 0)
    fm_np = np.random.randint(size=shape, dtype=feature.dtype, low = -16, high = 16)
    fm_nd = tvm.nd.array(fm_np, ctx)

    k_np = np.random.randint(size=kshape, dtype=kernel.dtype, low = -16, high = 16)
    k_nd = tvm.nd.array(k_np, ctx)

    res_nd = tvm.nd.array(np.zeros(imd4_shape, dtype=res_host.dtype), ctx)

    nnpu.set_dump(False)

    func(fm_nd, k_nd, res_nd)

    res_np = res_nd.asnumpy()

# calculate ground truth
feature = tvm.placeholder(shape, dtype_n, 'feature-map')
kernel = tvm.placeholder(kshape, dtype_n, 'kernel')
res_shape = imd4_shape  # (x, y, oc)
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