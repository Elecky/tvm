'''
convolution demo
====================
this demo first use im2col to convert input feature map, then use GEMM to do convolution.
'''
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
    #==================================#
    # ------ first define shapes ------
    #==================================#

    # the input image and kernel shapes.
    shape = (32, 18, 32)     # layout: HWC
    kshape = (3, 3, 64, 32)  # layout: HWOI
    # convolution stride.
    stride_h, stride_w = 1, 1
    
    # the shape of MAC. 
    # this determines the input tensor shape of GEMM instruction, ie (n_row, facotr) and (n_col, factor)
    gemm_shape = (8, 8, 8)
    n_row, factor, n_col = gemm_shape

    # do some checking on shapes. in this test we don't do padding, so shapes must meet some restrictions.
    assert shape[-1] == kshape[-1], 'feature map in-channel != kernel in-channel'
    assert shape[0] >= kshape[0] and shape[1] >= kshape[1], 'feature map smaller than kernel'
    assert shape[-1] % factor == 0, 'in-channel not divisible to factor'
    assert kshape[-2] % n_col == 0, 'out-channel not divisible to gemm insn NColOut'
    nvctr_unit = env.cfg['vector_unit']['size']

    # NNPU simulator has configuration on data types, so first read dtype config from config file.
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    # create input feature and kernel tensors.
    feature = tvm.placeholder(shape, dtype_n, 'feature-map')
    kernel = tvm.placeholder(kshape, dtype_n, 'kernel')

    # conv_shape is the final convolution output shape.
    # the layout: HWC.
    conv_shape = ( (shape[0] + 1 - kshape[0]) // stride_h, (shape[1] + 1 - kshape[1]) // stride_w, kshape[2])
    # we want to tile the feature map after im2col, so check whether the shape are divisible be tiling factor.
    assert conv_shape[1] % n_row == 0, \
            'can not tile feature map correctly, conv_output_shape width not divisible by GEMM n_row'

    # the shape of feature map after im2col and tiling.
    # after im2col, the shape should be (PH, PW, KH, KW, C). NOTE THAT we don't combine some of the dimensions, odinaryly, it should be (PH*PW, KH*KW*C).
    # PH and PW means numebrs of patches vertically and horizontally, a patch is simply a convolution window.
    # KH and KW means kernel height and width.
    # after tiling, the shape is (PH, PW.o, KH, KW, C.o, PW.i, C.i), where PW.i=n_row, C.i=factor.
    # the .o/.i suffix indicates outter and inner dimension after tiling.
    feature_shape_packed = (conv_shape[0], conv_shape[1] // n_row, kshape[0], kshape[1], 
                            shape[2] // factor, n_row, factor)
    # the shape of kernel after tiling. 
    # the layout: HWOIoi.
    kernel_shape_packed = (kshape[0], kshape[1], kshape[2] // n_col, kshape[3] // factor, n_col, factor)
    # the shape of GEMM output. 
    # it differs from conv_shape. since feature and kernel are tiled, the output is tiled too.
    conv_acc_shape = (conv_shape[0], conv_shape[1] // n_row, kshape[2] // n_col,
                      n_row, n_col)

    #=================================================================#
    # ------ after all shapes defined, begin compute describing. ------
    #=================================================================#
    # feature_buf is the feature map after im2col and tiling. 
    # feature_buf is on scratchpad, that is, im2col and tiling is done when loading to scratchpad.
    feature_buf = tvm.compute(feature_shape_packed,
                              lambda ph, pw_o, wh, ww, co, pw_i, ci: 
                                    feature[ph*stride_h + wh, (pw_o*n_row+pw_i)*stride_w+ww, co*factor+ci],
                              'feature_packed')
    # kernel_buf is the kernel after tiling. it's on scratchpad, tiling is done when loading to scratchpad.
    kernel_buf = tvm.compute(kernel_shape_packed,
                            lambda kh, kw, oco, co, oci, ci:
                                kernel[kh, kw, oco*n_col+oci, co*factor+ci],
                            'kernel_packed')

    # the reduce_axes of GEMM. they are reduction axis on kernel height/width and 2 tiled channel, respectively.
    kh_reduce = tvm.reduce_axis((0, kshape[0]), 'kh')
    kw_reduce = tvm.reduce_axis((0, kshape[1]), 'kw')
    co_reduce = tvm.reduce_axis((0, feature_shape_packed[4]), 'co')
    ci_reduce = tvm.reduce_axis((0, factor), 'ci')

    # the GEMM computation.
    # it is, hard to describe......
    conv_acc = tvm.compute(conv_acc_shape, 
                           lambda ph, pw_o, oco, pw_i, oci:
                                tvm.sum(feature_buf[ph, pw_o, kh_reduce, kw_reduce, co_reduce, pw_i, ci_reduce].astype(dtype_w) *
                                        kernel_buf[kh_reduce, kw_reduce, oco, co_reduce, oci, ci_reduce].astype(dtype_w), 
                                        axis=[kh_reduce, kw_reduce, co_reduce, ci_reduce]),
                           'conv_acc')

    # reshape output. 
    # this is done when copying from accumulation buffer to scratchpad.
    conv = tvm.compute(conv_shape,
                       lambda ph, pw, oc: conv_acc[ph, pw / n_row, oc / n_col, pw % n_row, oc % n_col],
                       'conv')
    # indentity computation.
    conv_host = tvm.compute(conv_shape, lambda *i: conv(*i))

    res_host = conv_host

    # ------ this ends the computation description. ------

    #==================================#    
    # ------ begin scheduling ------
    #==================================#

    # set the memory scopes of tensors that should be on accelerator.
    # here we put keature and kernel on buffer0 and buffer1 respectively.
    nnpu.utils.MarkScope(feature_buf, 'buffer1')
    nnpu.utils.MarkScope(kernel_buf, 'buffer2')
    # the GEMM output is on accumulation buffer.
    nnpu.utils.MarkScope(conv_acc, 'acc')
    nnpu.utils.MarkScope(conv, 'buffer3')

    s = nnpu.create_schedule(res_host.op)
    # reorder the GEMM compute stage.
    # the rule is, first make sure the axes of one GEMM instruction are the last 3 iterations, then other reduction axes follows.
    ph, pw_o, oco, pw_i, oci = conv_acc.op.axis
    s[conv_acc].reorder(ph, pw_o, kh_reduce, kw_reduce, co_reduce, oco, pw_i, oci ,ci_reduce)
    # tensorize
    s[conv_acc].tensorize(pw_i, env.intrins.get('GEMM', shape=gemm_shape, mode='inc', scope_in1='buffer1', scope_in2='buffer2',
                                              scope_out='acc'))
    # unroll some axes, to eliminate loop overhead.
    # s[conv_acc].unroll(co_reduce)
    # s[conv_acc].unroll(kw_reduce)
    # s[conv_acc].unroll(kh_reduce)

    # use compute_at to attach GEMM stage into CopyAcc2Buf stage(copying from accumulation buffer to scratchpad).
    # because accumulation buffer may be quiet small, so we copy output into scratchpad 
    #   once GEMM finished computing one row of output.
    s[conv_acc].compute_at(s[conv], s[conv].op.axis[0])

    # split copy, to eliminate division and modolar arithmetic.
    s[conv].tile(conv.op.axis[1], conv.op.axis[2], n_row, n_col)

    # split output
    ph, pw, oc = res_host.op.axis
    # factors is the size of output to generate every block
    factors = [15, 8, 64]
    assert factors[1] % n_row == 0 and factors[2] % n_col == 0, 'after split, still should be able to be tensorized'
    pho, phi = s[res_host].split(ph, factors[0])
    pwo, pwi = s[res_host].split(pw, factors[1])
    oco, oci = s[res_host].split(oc, factors[2])
    s[res_host].reorder(oco, pho, pwo, phi, pwi, oci)

    # compute_at
    s[conv].compute_at(s[res_host], s[res_host].leaf_iter_vars[2])
    s[kernel_buf].compute_at(s[res_host], oco)
    s[feature_buf].compute_at(s[res_host], pwo)

    # add copy pragma.
    s[feature_buf].pragma(feature_buf.op.axis[0], env.dma_copy_to_buf)
    s[kernel_buf].pragma(kernel_buf.op.axis[0], env.dma_copy_to_buf)
    s[res_host].pragma(phi, env.dma_copy_from_buf)
    s[conv].pragma(s[conv].leaf_iter_vars[-2], env.copy_acc2buf)
    #==================================#
    # ------ this ends the scheduling ------
    #==================================#

    print(nnpu.lower(s, [feature, kernel, res_host], simple_mode=True))

    # func = tvm.build(s, [feature, kernel, res_host], 'llvm', 'llvm', 'nnpu_conv')
    func = nnpu.build(s, [feature, kernel, res_host], 'nnpu', 'llvm', 'nnpu_conv')
    # print('------------------- device module 1 asm code: ')
    # print(func.imported_modules[0].get_source('asm'))
    print('------------------- device module 1 TVM IR: ')
    print(func.imported_modules[0].get_source('ir'))
    print('------------------- device module 1 uop: ')
    print(func.imported_modules[0].get_source('uop'))
    # exit()

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
                    tvm.sum(feature[x * stride_h + rx, y * stride_w + ry, rc].astype(dtype_w) * 
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