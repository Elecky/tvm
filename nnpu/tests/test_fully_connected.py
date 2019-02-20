import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

with ScheduleProcHelper():
    env = nnpu.get_env()
    nnpu.set_device(env, type='S0')

    out_channel = 32
    in_channel = 64
    gemm_shape = (16, 16, 1)  # the shape of gemm instruction      
    assert out_channel % gemm_shape[0] == 0, 'out_channel not divisble to gemm insn input1 row count'
    assert in_channel % gemm_shape[1] == 0, 'in_channel not divisble to gemm insn factor'  
    weight_shape = (out_channel, in_channel)
    data_shape = (in_channel, )
    bias_shape = (out_channel, )
    factor = gemm_shape[1]

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    weight = tvm.placeholder(weight_shape, dtype_n, 'w')
    data = tvm.placeholder(data_shape, dtype_n, 'd')
    bias = tvm.placeholder(bias_shape, dtype_w, 'bias')

    weight_buf, weight_dram = nnpu.utils.CopyHtoBuf(weight, 'a')
    data_buf, data_dram = nnpu.utils.CopyHtoBuf(data, 'b')
    bias_buf, _ = nnpu.utils.CopyHtoBuf(bias, 'bias')

    k = tvm.reduce_axis((0, in_channel), 'k0')
    prod_shape = (out_channel, )
    prod_buf = tvm.compute(prod_shape,
                        lambda i: tvm.sum(weight_buf[i, k].astype(dtype_w) * 
                                        data_buf[k].astype(dtype_w), axis=k),
                        'prod')
    nnpu.utils.MarkScope(prod_buf, 'acc')

    out_buf = nnpu.utils.CopyAccToBuf(prod_buf, 'out')
    
    res_buf = tvm.compute((out_channel, ),
                        lambda i: out_buf[i] + bias_buf[i], 'res')
    nnpu.utils.MarkScope(res_buf)
    res_host, _ = nnpu.utils.CopyBufToH(res_buf, 'res')
    
    s = nnpu.utils.create_schedule(res_host.op)

    xo, xi = s[prod_buf].split(prod_buf.op.axis[0], factor=gemm_shape[0])
    ro, ri = s[prod_buf].split(prod_buf.op.reduce_axis[0], factor=factor)
    s[prod_buf].reorder(xo, ro, xi, ri)
    s[prod_buf].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, 
                                    mode='inc', reduce=True, scope_out='acc'))

    # you can move gemm into acc2buffer copy, but manual pragma is needed.
    #xo, xi = s[out_buf].split(out_buf.op.axis[0], factor=gemm_shape[0])
    #s[prod_buf].compute_at(s[out_buf], xo)
    #s[out_buf].pragma(xi, env.copy_acc2buf)

    xo, xi = s[res_buf].split(res_buf.op.axis[0], factor=16)
    s[res_buf].tensorize(xi, env.intrins.get('VAddV', mode='w'))

    print(nnpu.lower(s, [weight, data, bias, res_host], simple_mode=True))

    func = nnpu.build(s, [weight, data, bias, res_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=weight_shape, dtype=weight.dtype, low = -32, high = 32)
    a_nd = tvm.nd.array(a_np, ctx)
    d_np = np.random.randint(size=data_shape, dtype=data.dtype, low = -32, high = 32)
    d_nd = tvm.nd.array(d_np, ctx)
    b_np = np.random.randint(size=bias_shape, low=-5000, high=5000, dtype=bias.dtype)
    b_nd = tvm.nd.array(b_np, ctx)

    out_nd = tvm.nd.array(np.zeros((out_channel, ), dtype=res_host.dtype), ctx)

    func(a_nd, d_nd, b_nd, out_nd)

    print(out_nd.asnumpy())
    gt = np.dot(a_np, d_np.astype(res_host.dtype))
    gt = gt + b_np
    print('numpy result = ')
    print(gt)
    np.testing.assert_allclose(out_nd.asnumpy(), gt)
    print('test passed')