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
parser.add_argument('--cfg', type=int, help='the config file to use', 
                    default=0, choices=[0, 1, 2, 3, 4])
parser.add_argument('--profile', type=bool, help='enable profiling', 
                    default=True)
args = parser.parse_args()

if (args.profile):
    profile_dir = '/home/jian/Documents/nnpu_profile'
    nnpu.set_profile(['timeline', 'memory_access_latency'], profile_dir)

config = args.cfg
cfg_path = './nnpu_config.{0}.yaml'.format(config)
if (config >= 0):
    gemm_shape = (8, 8, 8)
    factors = [8, 16, 128]
if (config >= 1):
    gemm_shape = (16, 16, 16)
    factors = [8, 16, 128]
if (config >= 2):
    pass
if (config >= 3):
    factors = [8, 16, 256]
if (config >= 4):
    pass

def packed_conv2d(data, kernel, bias, strides, padding, out_dtype):
    
    env = nnpu.get_env()
    assert isinstance(strides, int) or len(strides) == 2

    filter_height, filter_width, num_filter, in_channel = [topi.util.get_const_int(x) for x in kernel.shape]

    # TODO: this is for demo only
    # batch_size, in_height, in_width, in_channel = [topi.util.get_const_int(x) for x in data.shape]
    # if (in_height == 28):
    #     padding = (padding[0] + 2, padding[1] + 2)

    if isinstance(strides, int):
        stride_height = stride_width = strides
    else:
        stride_height, stride_width = strides
    out_channel = num_filter
    if(padding[0]):
        pad_data = topi.nn.pad(data, [0, padding[0], padding[1], 0], name = "pad_data")
    else:
        pad_data = data 
    batch_size, in_height, in_width, in_channel = [topi.util.get_const_int(x) for x in pad_data.shape]

    out_height = topi.util.simplify((in_height - filter_height) // stride_height + 1)
    out_height = topi.util.get_const_int(out_height)

    out_width = topi.util.simplify((in_width - filter_width) // stride_width + 1)
    out_width = topi.util.get_const_int(out_width)

    macops = out_height * out_width * num_filter * filter_height * filter_width * in_channel

    k_f_h = tvm.reduce_axis((0, filter_height))
    k_f_w = tvm.reduce_axis((0, filter_width))
    k_ci_o = tvm.reduce_axis((0, in_channel // gemm_shape[1]))
    k_ci_i = tvm.reduce_axis((0, gemm_shape[1]))
    if (bias is None):
        col_data_shape = (batch_size, out_height, out_width, filter_height, filter_width, in_channel)
        col_data = tvm.compute(col_data_shape,
                            lambda n, h, w, kh, kw, c: pad_data[n, h + kh*stride_height, w+kw*stride_width, c],
                            name='col_data')

        assert col_data_shape[-1] % gemm_shape[1] == 0, 'not aligned'
        assert col_data_shape[2] % gemm_shape[0] == 0, 'not aligned'
        # Nz layout
        packed_data_shape = (batch_size, out_height, filter_height, filter_width, in_channel // gemm_shape[1], out_width, gemm_shape[1])
        packed_data = tvm.compute(packed_data_shape,
                            lambda n, h, kh, kw, co, w, ci: col_data[n, h, w, kh, kw, co * gemm_shape[1] + ci],
                            name='packed_data')
        # Nz layout
        packed_kernel_shape = (filter_height, filter_width, out_channel // gemm_shape[2], in_channel // gemm_shape[1], gemm_shape[2], gemm_shape[1])
        packed_kernel = tvm.compute(packed_kernel_shape,
                            lambda kh, kw, oco, co, oci, ci: kernel[kh, kw, oco*gemm_shape[2]+oci, co*gemm_shape[1]+ci],
                            name='packed_kernel')
        
        res1 = tvm.compute((batch_size, out_height, out_channel // gemm_shape[2], out_width, gemm_shape[2]),
                        lambda n, h, oco, w, oci: 
                            tvm.sum(packed_data[n, h, k_f_h, k_f_w, k_ci_o, w, k_ci_i]
                                * packed_kernel[k_f_h, k_f_w, oco, k_ci_o, oci, k_ci_i],
                                axis=[k_f_h, k_f_w, k_ci_o, k_ci_i] ),
                        name='conv2d_inner')
        res = tvm.compute((batch_size, out_height, out_width, out_channel),
                        lambda n, h, w, oc: res1[n, h, oc / gemm_shape[2], w, oc % gemm_shape[2]],
                        name='conv2d_res', tag='packed_conv2d')
    else:
        col_data_shape = (batch_size, out_height, out_width, filter_height, filter_width, in_channel)
        col_data = tvm.compute(col_data_shape,
                            lambda n, h, w, kh, kw, c: pad_data[n, h + kh*stride_height, w+kw*stride_width, c],
                            name='col_data')

        assert col_data_shape[-1] % gemm_shape[1] == 0, 'not aligned'
        assert col_data_shape[2] % gemm_shape[0] == 0, 'not aligned'
        # Nz layout
        packed_data_shape = (batch_size, out_height, filter_height, filter_width, in_channel // gemm_shape[1], out_width, gemm_shape[1])
        packed_data = tvm.compute(packed_data_shape,
                            lambda n, h, kh, kw, co, w, ci: col_data[n, h, w, kh, kw, co * gemm_shape[1] + ci],
                            name='packed_data')
        # Nz layout
        packed_kernel_shape = (filter_height, filter_width, out_channel // gemm_shape[2], in_channel // gemm_shape[1], gemm_shape[2], gemm_shape[1])
        packed_kernel = tvm.compute(packed_kernel_shape,
                            lambda kh, kw, oco, co, oci, ci: kernel[kh, kw, oco*gemm_shape[2]+oci, co*gemm_shape[1]+ci],
                            name='packed_kernel')
        
        res1 = tvm.compute((batch_size, out_height, out_channel // gemm_shape[2], out_width, gemm_shape[2]),
                        lambda n, h, oco, w, oci: 
                            tvm.sum(packed_data[n, h, k_f_h, k_f_w, k_ci_o, w, k_ci_i]
                                * packed_kernel[k_f_h, k_f_w, oco, k_ci_o, oci, k_ci_i],
                                axis=[k_f_h, k_f_w, k_ci_o, k_ci_i] ),
                        name='conv2d_inner')
        bias_res = tvm.compute((batch_size, out_height, out_channel // gemm_shape[2], out_width, gemm_shape[2]),
                        lambda n, h, oco, w, oci: res1[n, h, oco, w, oci] + bias[oco * gemm_shape[2] + oci],
                        name='bias_res')
        res = tvm.compute((batch_size, out_height, out_width, out_channel),
                        lambda n, h, w, oc: bias_res[n, h, oc / gemm_shape[2], w, oc % gemm_shape[2]],
                        name='conv2d_res', tag='packed_conv2d')

    return res, macops

with ScheduleProcHelper(), nnpu.Environment(cfg_path):
    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    #==================================#
    # ------ first define shapes ------
    #==================================#

    # the input image and kernel shapes.
    shape = (1, 16, 16, 128)     # layout: HWC
    kshape = (3, 3, 256, 128)  # layout: HWOI

    # convolution stride.
    stride_h, stride_w = 1, 1
    
    # the shape of MAC. 
    # this determines the input tensor shape of GEMM instruction, ie (n_row, facotr) and (n_col, factor)
    n_row, factor, n_col = gemm_shape

    # do some checking on shapes. in this test we don't do padding, so shapes must meet some restrictions.
    assert shape[-1] == kshape[-1], 'feature map in-channel != kernel in-channel'
    assert shape[1] >= kshape[0] and shape[2] >= kshape[1], 'feature map smaller than kernel'
    assert shape[-1] % factor == 0, 'in-channel not divisible to factor'
    assert kshape[-2] % n_col == 0, 'out-channel not divisible to gemm insn NColOut'
    nvctr_unit = env.cfg['vector_unit']['size']

    # NNPU simulator has configuration on data types, so first read dtype config from config file.
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    # create input feature and kernel tensors.
    feature = tvm.placeholder(shape, dtype_n, 'feature-map')
    kernel = tvm.placeholder(kshape, dtype_n, 'kernel')
    bias = tvm.placeholder((kshape[2], ), dtype_n, 'bias')

    output, macops = packed_conv2d(feature, kernel, bias, (stride_h, stride_w), (1, 1), 'int8')

    out_shape = [topi.util.get_const_int(x) for x in output.shape]

    #==================================#    
    # ------ begin scheduling ------
    #==================================#
    s = tvm.create_schedule(output.op)

    N, OH, OW, OC = [topi.util.get_const_int(x) for x in output.shape]

    conv2d_res = output

    # bias_res = s.cache_write(output, env.get_scope('buffer4'))
    bias_res = output.op.input_tensors[0]
    conv2d_res, bias = bias_res.op.input_tensors

    packed_data, packed_kernel = conv2d_res.op.input_tensors
    col_data = packed_data.op.input_tensors[0]
    pad_data = col_data.op.input_tensors[0]
    kernel = packed_kernel.op.input_tensors[0]

    # cache read data/kernel
    if isinstance(pad_data.op, tvm.tensor.ComputeOp) and "pad" in pad_data.op.tag:
        data = pad_data.op.input_tensors[0]
    else:
        data = pad_data
        pad_data = None
    if (bias is not None):
        cbias = s.cache_read(bias, env.get_scope('buffer5'), [bias_res])

    # set scopes
    s[packed_data].set_scope(env.get_scope('buffer1'))
    s[packed_kernel].set_scope(env.get_scope('buffer2'))
    s[conv2d_res].set_scope(env.get_scope('buffer3'))
    s[bias_res].set_scope(env.get_scope('buffer4'))
    # s[cbias].set_scope(env.get_scope('buffer5'))

    # split output
    hfactor, wfactor, ocfactor = factors
    if (OH % (hfactor * 2) != 0):
        hfactor = 4
    # if (OW % wfactor != 0):
    #     wfactor = 8
    if (OC < ocfactor):
        ocfactor = OC
    assert OC % ocfactor == 0, 'split factor error'
    if (OH // hfactor < 2):
        hfactor = OH // 2
    print('hfactor, wfactor, ocfactor = ', hfactor, wfactor, ocfactor)

    ho, hi = s[output].split(output.op.axis[1], hfactor)
    wo, wi = s[output].split(output.op.axis[2], wfactor)
    oco, oci = s[output].split(output.op.axis[3], ocfactor)
    oci0, oci1 = s[output].split(oci, gemm_shape[2])  # to eliminate division
    cthread_h, ho = s[output].split(ho, nparts=2)
    s[output].reorder(cthread_h, ho, wo, oco, hi, wi, oci0, oci1)
    out_oco = oco

    s[output].bind(cthread_h, tvm.thread_axis('cthread'))

    s[cbias].compute_at(s[output], cthread_h)

    # pragma dma copy
    s[packed_data].pragma(packed_data.op.axis[0], env.dma_copy_to_buf)
    s[packed_kernel].pragma(packed_kernel.op.axis[0], env.dma_copy_to_buf)
    s[cbias].pragma(cbias.op.axis[0], env.dma_copy_to_buf)
    s[output].pragma(wi, env.dma_copy_from_buf)

    # split and tensorize bias_res
    wo, wt = s[bias_res].split(bias_res.op.axis[3], gemm_shape[2])
    s[bias_res].reorder(*bias_res.op.axis[0:3], wo, wt, bias_res.op.axis[4])
    # tensorize on wt
    s[bias_res].pragma(wt, 'nnpu.vector', str({'code': 'matrix-vector', 'shape': (gemm_shape[0], gemm_shape[2])}))
    s[bias_res].compute_at(s[output], hi)

    # split and tensorize conv2d_res
    n, h, oco, w, oci = conv2d_res.op.axis
    kfh, kfw, kcio, kcii = conv2d_res.op.reduce_axis
    wo, wt = s[conv2d_res].split(w, gemm_shape[2])
    in_channel_block = topi.util.get_const_int(packed_data.shape[4])
    factor = 4 if in_channel_block % 4 == 0 else 1
    kcioo, kcioi = s[conv2d_res].split(kcio, factor)
    s[conv2d_res].reorder(n, kfh, kfw, kcioo, kcioi, h, oco, wo, wt, oci, kcii)
    # tensorize on wt
    s[conv2d_res].tensorize(wt, env.intrins.get('GEMM', shape=gemm_shape, scope_in1='buffer1', scope_in2='buffer2', scope_out='buffer3', mode='n'))
    s[conv2d_res].compute_at(s[output], out_oco)

    # compute_at data and kernel
    s[packed_data].compute_at(s[conv2d_res], kcioo)
    s[col_data].compute_inline()
    s[packed_kernel].compute_at(s[conv2d_res], kcioo)

    # print(tvm.lower(s, [data, kernel, bias, output], simple_mode=True))
    # lib = tvm.build(s, [data, kernel, bias, output], 'nnpu', 'llvm', 'func')
    # print(lib.imported_modules[0].get_source('ir'))

    #==================================#
    # ------ this ends the scheduling ------
    #==================================#

    print(nnpu.lower(s, [feature, kernel, bias, output], simple_mode=True))

    # func = tvm.build(s, [feature, kernel, res_host], 'llvm', 'llvm', 'nnpu_conv')
    func = nnpu.build(s, [feature, kernel, bias, output], 'nnpu', 'llvm', 'nnpu_conv')
    # print('------------------- device module 1 asm code: ')
    # print(func.imported_modules[0].get_source('asm'))
    print('------------------- device module 1 TVM IR: ')
    print(func.imported_modules[0].get_source('ir'))
    # print('------------------- device module 1 uop: ')
    # print(func.imported_modules[0].get_source('uop'))
    # exit()

    ctx = tvm.nd.TVMContext(13, 0)
    fm_np = np.random.randint(size=shape, dtype=feature.dtype, low = -16, high = 16)
    fm_nd = tvm.nd.array(fm_np, ctx)

    k_np = np.random.randint(size=kshape, dtype=kernel.dtype, low = -16, high = 16)
    k_nd = tvm.nd.array(k_np, ctx)

    bias_np = np.random.randint(size=(kshape[2],), dtype=kernel.dtype, low = -16, high = 16)
    bias_nd = tvm.nd.array(bias_np, ctx)

    res_nd = tvm.nd.array(np.zeros(out_shape, dtype=output.dtype), ctx)

    nnpu.set_dump(False)

    func(fm_nd, k_nd, bias_nd, res_nd)
    print('execute finish')

    res_np = res_nd.asnumpy()

# calculate ground truth
feature = tvm.placeholder(shape, dtype_n, 'feature-map')
pad_data = topi.nn.pad(feature, [0, 1, 1, 0], name = "pad_data")
kernel = tvm.placeholder(kshape, dtype_n, 'kernel')
bias = tvm.placeholder((kshape[2], ), dtype_n, 'bias')

res_shape = out_shape  # (x, y, oc)
rc = tvm.reduce_axis((0, shape[-1]), 'rc')
ry = tvm.reduce_axis((0, kshape[1]), 'ry')
rx = tvm.reduce_axis((0, kshape[0]), 'rx')

res_innner = tvm.compute(res_shape, 
                lambda n, x, y, oc: 
                    tvm.sum(pad_data[n, x * stride_h + rx, y * stride_w + ry, rc] * 
                            kernel[rx, ry, oc, rc], 
                            axis=[rx, ry, rc]),
                'res')
res = tvm.compute(res_shape, lambda n, x, y, oc: res_innner[n, x, y, oc] + bias[oc], 'output')
s1 = tvm.create_schedule(res.op)
#print(tvm.lower(s1, [feature, kernel, res], simple_mode=True))
h_func = tvm.build(s1, [feature, kernel, bias, res], 'llvm', 'llvm', 'host_conv')

fm_nd = tvm.nd.array(fm_np)
k_nd = tvm.nd.array(k_np)
bias_nd = tvm.nd.array(bias_np)

gt_nd = tvm.nd.array(np.zeros(res_shape, dtype=dtype_n))
h_func(fm_nd, k_nd, bias_nd, gt_nd)

np.testing.assert_allclose(res_nd.asnumpy(), gt_nd.asnumpy())
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