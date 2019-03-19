import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

with ScheduleProcHelper():
    env = nnpu.get_env()
    nnpu.set_device(env, type='SC')
    shape = (48, 64)  # (32, 32) reshaped to (32, 2, 16)
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

    k = tvm.reduce_axis((0, factor), 'k0')
    prod_shape = (shape[1] / factor, shape[0])
    prod_buf = tvm.compute(prod_shape,
                        lambda i, j: tvm.sum(a_buf[j, i * factor + k].astype(dtype_w) * 
                                        b_buf[i * factor + k].astype(dtype_w), axis=k),
                        'prod')
    nnpu.utils.MarkScope(prod_buf)
    #prod_host, _ = nnpu.utils.CopyBufToH(prod_buf, 'prod')

    k = tvm.reduce_axis((0, prod_shape[0]), 'k1')
    out_buf = tvm.compute((shape[0], ), 
                        lambda i: tvm.sum(prod_buf[k, i], axis=k), 'out_buf')
    nnpu.utils.MarkScope(out_buf)
    out_host, _ = nnpu.utils.CopyBufToH(out_buf, 'out')
    
    s = nnpu.utils.create_schedule(out_host.op)

    yo, yi = s[prod_buf].split(prod_buf.op.axis[1], factor=gemm_shape[0])
    s[prod_buf].reorder(prod_buf.op.axis[0], yo, yi, prod_buf.op.reduce_axis[0])
    s[prod_buf].tensorize(yi, env.intrins.get('GEMM', shape=gemm_shape, mode='inc', reduce=True))

    xo, xi = s[out_buf].split(out_buf.op.axis[0], factor=env.cfg['vector_unit']['size'])
    ro, ri = s[out_buf].split(out_buf.op.reduce_axis[0], factor=1)
    s[out_buf].reorder(xo, ro, ri, xi)
    s[out_buf].tensorize(ri, env.intrins.get('VAddMerge', mode='w'))

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