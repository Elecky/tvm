import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder((16, ), dtype_n, 'a')
    b = tvm.placeholder((16, ), dtype_n, 'b')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)
    
    c_buf = tvm.compute((16, ), lambda i: a_buf[i] + b_buf[i], 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    mul_buf = tvm.compute((16, ), 
                lambda i: a_buf[i].astype(dtype_w) * b_buf[i].astype(dtype_w), 'mul_buf')
    sph.MarkScope(mul_buf)
    mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    gtm_buf = tvm.compute((16, ), 
                lambda i: tvm.select(a_buf[i] > b_buf[i], a_buf[i], b_buf[i]), 'gtm_buf')
    sph.MarkScope(gtm_buf)
    gtm_host, gtm_dram = nnpu.utils.CopyBufToH(gtm_buf, 'gtm', sph)

    s = tvm.create_schedule([c_host.op, mul_host.op, gtm_host.op])
    sph.Transform(s)
    s[c_buf].tensorize(s[c_buf].op.axis[0], env.intrins.get('VAddV', mode='n'))
    s[mul_buf].tensorize(s[mul_buf].op.axis[0], env.intrins.get('VMulV', mode='inc'))
    s[gtm_buf].tensorize(s[gtm_buf].op.axis[0], env.intrins.get('VGTMV', mode='n'))

    print(nnpu.lower(s, [a, b, c_host, mul_host, gtm_host], simple_mode=True))
    func = nnpu.build(s, [a, b, c_host, mul_host, gtm_host], 'nnpu', 'llvm', name='nnpu_exp')


    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(16, ), dtype=a.dtype, low = -64, high = 63)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=(16, ), dtype=b.dtype, low = -64, high = 63)    
    b_nd = tvm.nd.array(b_np, ctx)
    
    c_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros((16, )).astype(mul_host.dtype), ctx)
    gtm_nd = tvm.nd.array(np.zeros((16, )).astype(gtm_host.dtype), ctx)

    func(a_nd, b_nd, c_nd, mul_nd, gtm_nd)
    print('a = ')
    print(a_np)
    print('b = ')
    print(b_np)
    print('a + b =')
    print(c_nd.asnumpy())
    print("numpy ground truth is")
    print(a_np + b_np)
    print('(int16)a * b =')
    print(mul_nd.asnumpy())
    np.testing.assert_allclose(mul_nd.asnumpy(), np.multiply(a_np, b_np, dtype=mul_host.dtype))
    print('max(a, b) = ')
    print(gtm_nd.asnumpy())

if __name__ == '__main__':
    test()