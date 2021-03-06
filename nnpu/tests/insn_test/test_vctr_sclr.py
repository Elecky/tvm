import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder((64, ), dtype_n, 'a')
    b = tvm.placeholder((1, ), dtype_n, 'b')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)
    
    c_buf = tvm.compute((64, ), lambda i: a_buf[i] + b_buf[0], 'c_buf')
    sph.MarkScope(c_buf)
    c_host, _ = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    sub_buf = tvm.compute((64, ), lambda i: a_buf[i] - b_buf[0], 'sub_buf')
    sph.MarkScope(sub_buf)
    sub_host, _ = nnpu.utils.CopyBufToH(sub_buf, 'sub', sph)
    
    rsub_buf = tvm.compute((64, ), lambda i: b_buf[0] - a_buf[i], 'rsub_buf')
    sph.MarkScope(rsub_buf)
    rsub_host, _ = nnpu.utils.CopyBufToH(rsub_buf, 'rsub', sph)

    mul_buf = tvm.compute((64, ), 
                lambda i: a_buf[i].astype(dtype_w) * b_buf[0].astype(dtype_w), 'mul_buf')
    sph.MarkScope(mul_buf)
    mul_host, _ = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    div_buf = tvm.compute((64, ), lambda i: a_buf[i] / b_buf[0], 'div_buf')
    sph.MarkScope(div_buf)
    div_host, _ = nnpu.utils.CopyBufToH(div_buf, 'div', sph)

    rdiv_buf = tvm.compute((64, ), lambda i: b_buf[0] / a_buf[i], 'rdiv_buf')
    sph.MarkScope(rdiv_buf)
    rdiv_host, _ = nnpu.utils.CopyBufToH(rdiv_buf, 'rdiv', sph)

    gtm_buf = tvm.compute((64, ), 
                lambda i: tvm.max(a_buf[i], b_buf[0]), 'gtm_buf')
    sph.MarkScope(gtm_buf)
    gtm_host, gtm_dram = nnpu.utils.CopyBufToH(gtm_buf, 'gtm', sph)

    s = tvm.create_schedule([c_host.op, sub_host.op, mul_host.op, rsub_host.op, div_host.op,
                             rdiv_host.op, gtm_host.op])
    sph.Transform(s)
    xo, xi = s[c_buf].split(c_buf.op.axis[0], 16)
    s[c_buf].tensorize(xi, env.intrins.get('VAddS', mode='n'))
    xo, xi = s[sub_buf].split(sub_buf.op.axis[0], 16)
    s[sub_buf].tensorize(xi, env.intrins.get('VSubS', mode='n'))
    xo, xi = s[rsub_buf].split(rsub_buf.op.axis[0], 16)
    s[rsub_buf].tensorize(xi, env.intrins.get('SSubV', mode='n'))
    xo, xi = s[mul_buf].split(mul_buf.op.axis[0], 16)
    s[mul_buf].tensorize(xi, env.intrins.get('VMulS', mode='inc'))
    xo, xi = s[div_buf].split(div_buf.op.axis[0], 16)
    s[div_buf].tensorize(xi, env.intrins.get('VDivS', mode='n'))
    xo, xi = s[rdiv_buf].split(rdiv_buf.op.axis[0], 16)
    s[rdiv_buf].tensorize(xi, env.intrins.get('SDivV', mode='n'))
    xo, xi = s[gtm_buf].split(gtm_buf.op.axis[0], 16)
    s[gtm_buf].tensorize(xi, env.intrins.get('VGTMS', mode='n'))

    print(nnpu.lower(s, [a, b, c_host, sub_host, mul_host, rsub_host, div_host, rdiv_host, gtm_host], 
            simple_mode=True))
    func = nnpu.build(s, [a, b, c_host, sub_host, mul_host, rsub_host, div_host, rdiv_host, gtm_host], 
            'nnpu', 'llvm', name='nnpu_exp')

    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(64, ), dtype=a.dtype, low = 1, high = 63)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=(1, ), dtype=b.dtype, low = 2, high = 31)    
    b_nd = tvm.nd.array(b_np, ctx)
    
    c_nd = tvm.nd.array(np.zeros((64, )).astype(c_host.dtype), ctx)
    sub_nd = tvm.nd.array(np.zeros((64, )).astype(sub_host.dtype), ctx)
    rsub_nd = tvm.nd.array(np.zeros((64, )).astype(rsub_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros((64, )).astype(mul_host.dtype), ctx)
    div_nd = tvm.nd.array(np.zeros((64, )).astype(div_host.dtype), ctx)
    rdiv_nd = tvm.nd.array(np.zeros((64, )).astype(rdiv_host.dtype), ctx)
    gtm_nd = tvm.nd.array(np.zeros((64, )).astype(gtm_host.dtype), ctx)

    print('------------------- device module 1 llvm IR: ')
    print(func.imported_modules[0].get_source('ll'))

    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))

    func(a_nd, b_nd, c_nd, sub_nd, mul_nd, rsub_nd, div_nd, rdiv_nd, gtm_nd)
    print('a = ')
    print(a_np)
    print('b = ')
    print(b_np)
    print('a + b =')
    print(c_nd.asnumpy())
    print('numpy ground truth =')
    gt = a_np + b_np
    print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)
    print('a - b =')
    print(sub_nd.asnumpy())
    np.testing.assert_allclose(sub_nd.asnumpy(), a_np - b_np)

    print('b - a =')
    print(rsub_nd.asnumpy())
    np.testing.assert_allclose(rsub_nd.asnumpy(), b_np - a_np)

    print('a * b =')
    print(mul_nd.asnumpy())
    np.testing.assert_allclose(mul_nd.asnumpy(), a_np * b_np.astype(dtype_w))

    print('a / b =')
    print(div_nd.asnumpy())
    # numpy always round down, while in c, the numerator will be rounded to zero.
    #np.testing.assert_allclose(div_nd.asnumpy(), a_np / b_np)

    print('b / a =')
    print(rdiv_nd.asnumpy())

    print('max(a, b)=')
    print(gtm_nd.asnumpy())

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test of NNPU Op')
    parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
    args = parser.parse_args()
    with nnpu.Environment('./nnpu_config.yaml'):
        env = nnpu.get_env()
        nnpu.set_device(env, type=args.sim)
        test()