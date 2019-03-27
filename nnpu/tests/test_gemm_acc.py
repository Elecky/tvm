import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

with ScheduleProcHelper():
    env = nnpu.get_env()
    nnpu.set_device(env, type='S0')
    shape = (32, 32)  # (32, 32) reshaped to (32, 2, 16)
    gemm_shape = (16, 16, 1)
    factor = gemm_shape[1]
    assert shape[1] % factor == 0, 'emmmmmm'
    assert shape[0] % gemm_shape[0] == 0, 'well~~'
    assert shape[0] % env.cfg['vector_unit']['size'] == 0, 'oh~'

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder(shape, dtype_n, 'a')
    b = tvm.placeholder((shape[1], ), dtype_n, 'b')  # reshaped to (2, 16)

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a')
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b')

    k = tvm.reduce_axis((0, shape[1]), 'k0')
    prod_buf = tvm.compute((shape[0], ), 
                    lambda i: tvm.sum(a_buf[i, k].astype(dtype_w) * b_buf[k].astype(dtype_w),
                                      axis=k), 
                    'prod_buf')
    nnpu.utils.MarkScope(prod_buf, 'acc')

    out_buf = nnpu.utils.CopyAccToBuf(prod_buf, 'res')
    #nnpu.utils.MarkScope(out_buf)

    out_host, _ = nnpu.utils.CopyBufToH(out_buf, 'out')
    
    s = nnpu.utils.create_schedule(out_host.op)
    print(tvm.lower(s, [a, b, out_host], simple_mode=True))
    
    xo, xi = s[prod_buf].split(prod_buf.op.axis[0], factor=gemm_shape[0])
    ro, ri = s[prod_buf].split(prod_buf.op.reduce_axis[0], factor=factor)
    #ri = prod_buf.op.reduce_axis[0]
    s[prod_buf].reorder(xo, ro, xi, ri)
    print(tvm.lower(s, [a, b, out_host], simple_mode=True))
    s[prod_buf].tensorize(xi, env.intrins.get('GEMM', shape=gemm_shape, reduce=True, mode='inc',
                                              scope_out='acc'))
    print(nnpu.lower(s, [a, b, out_host], simple_mode=True))
    
    func = nnpu.build(s, [a, b, out_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape, dtype=a.dtype, low = -32, high = 32)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=(shape[1], ), dtype=b.dtype, low = -32, high = 32)
    b_nd = tvm.nd.array(b_np, ctx)

    out_nd = tvm.nd.array(np.zeros(shape[0], dtype=out_host.dtype), ctx)

    func(a_nd, b_nd, out_nd)

    print(out_nd.asnumpy())
    gt = np.dot(a_np, b_np.astype(out_host.dtype))
    print(gt)
    np.testing.assert_allclose(out_nd.asnumpy(), gt)
    print('test passed')