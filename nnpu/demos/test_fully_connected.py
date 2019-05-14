'''
dense layer demo
====================
this demo is a single batch dense layer inference
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

    out_channel = 256
    in_channel = 256
    # the MAC size, also the shape of gemm instruction.
    gemm_shape = (16, 16, 1)
    assert out_channel % gemm_shape[0] == 0, 'out_channel not divisble to gemm insn input1 row count'
    assert in_channel % gemm_shape[1] == 0, 'in_channel not divisble to gemm insn factor'  
    # weight layout: OI
    weight_shape = (out_channel, in_channel)
    # single batch input data shape:
    data_shape = (in_channel, )
    # bias vector shape:
    bias_shape = (out_channel, )
    factor = gemm_shape[1]

    # define tensors.
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    weight = tvm.placeholder(weight_shape, dtype_n, 'w')
    data = tvm.placeholder(data_shape, dtype_n, 'd')
    bias = tvm.placeholder(bias_shape, dtype_w, 'bias')

    #=================================================================#
    # ------ after all shapes defined, begin compute describing. ------
    #=================================================================#

    # first define copy compute of all tensors. some helper functions are used.
    # in this demo, data and weight are not tiled, so simply copy them.
    weight_buf, weight_dram = nnpu.utils.CopyHtoBuf(weight, 'a')
    data_buf, data_dram = nnpu.utils.CopyHtoBuf(data, 'b')
    bias_buf, _ = nnpu.utils.CopyHtoBuf(bias, 'bias')

    # the matrix multiply stage, we don't do any tiling on weight or data, so this is simply ordinary matrix multiply.
    # the result of without tiling is that, input tensors of GEMM instruction are not continous, 
    #   which may lead to unbiased access to scratchpad banks.
    # and at here, promote the data type from dtype_n to dtype_w.
    k = tvm.reduce_axis((0, in_channel), 'k0')
    prod_shape = (out_channel, )
    prod_buf = tvm.compute(prod_shape,
                           lambda i: tvm.sum(weight_buf[i, k].astype(dtype_w) * data_buf[k].astype(dtype_w), 
                                             axis=k),
                           'prod')
    # copy from accumulation buffer to scratchpad.
    out_buf = nnpu.utils.CopyAccToBuf(prod_buf, 'out')
    # add bias.
    res_buf = tvm.compute((out_channel, ),
                        lambda i: out_buf[i] + bias_buf[i], 'res')
    # copy back to host.
    res_host, _ = nnpu.utils.CopyBufToH(res_buf, 'res')
    
    # ------ this ends the computation description. ------

    #==================================#    
    # ------ begin scheduling ------
    #==================================#
    # set memory scopes.
    nnpu.utils.MarkScope(prod_buf, 'acc')
    nnpu.utils.MarkScope(res_buf)

    s = nnpu.utils.create_schedule(res_host.op)

    # split and reorder(or equally, tile) the matrix multiply stage.
    # because GEMM instruction only accepts (16, 16)*(16,) inputs,
    # we have to split the axes of GEMM stage, and reorder axes to make dimensions which describes matrix multiply of one tile at the last.
    xo, xi = s[prod_buf].split(prod_buf.op.axis[0], factor=gemm_shape[0])
    ro, ri = s[prod_buf].split(prod_buf.op.reduce_axis[0], factor=factor)
    s[prod_buf].reorder(xo, ro, xi, ri)
    # tensorize GEMM.
    s[prod_buf].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, 
                                    mode='inc', reduce=True, scope_out='acc'))

    # we can move gemm into acc2buffer copy.
    xo, xi = s[out_buf].split(out_buf.op.axis[0], factor=gemm_shape[0])
    s[prod_buf].compute_at(s[out_buf], xo)
    s[out_buf].pragma(xi, env.copy_acc2buf)

    # split and tensorize VAddV.
    nvctr_unit = env.cfg['vector_unit']['size']
    xo, xi = s[res_buf].split(res_buf.op.axis[0], factor=nvctr_unit)
    s[res_buf].tensorize(xi, env.intrins.get('VAddV', mode='w'))
    #==================================#
    # ------ this ends the scheduling ------
    #==================================#

    print(nnpu.lower(s, [weight, data, bias, res_host], simple_mode=True))

    func = nnpu.build(s, [weight, data, bias, res_host], 'nnpu', 'llvm', name='nnpu_func')
    # print('------------------- device module 1 asm code: ')
    # print(func.imported_modules[0].get_source('asm'))

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=weight_shape, dtype=weight.dtype, low = -32, high = 32)
    a_nd = tvm.nd.array(a_np, ctx)
    d_np = np.random.randint(size=data_shape, dtype=data.dtype, low = -32, high = 32)
    d_nd = tvm.nd.array(d_np, ctx)
    b_np = np.random.randint(size=bias_shape, low=-5000, high=5000, dtype=bias.dtype)
    b_nd = tvm.nd.array(b_np, ctx)

    out_nd = tvm.nd.array(np.zeros((out_channel, ), dtype=res_host.dtype), ctx)

    func(a_nd, d_nd, b_nd, out_nd)

    # print(out_nd.asnumpy())
    gt = np.dot(a_np, d_np.astype(res_host.dtype))
    gt = gt + b_np
    # print('numpy result = ')
    # print(gt)
    np.testing.assert_allclose(out_nd.asnumpy(), gt)
    print('test passed')