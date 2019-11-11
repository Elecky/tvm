import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    shape = (4, 64)
    # nvctr_unit = env.cfg['vector_unit']['size']
    nvctr_unit = 64
    # assert shape[0] % nvctr_unit == 0, 'error'

    a = tvm.placeholder(shape, dtype_n, 'a')
    b = tvm.placeholder(shape, dtype_n, 'b')
    
    sph = ScheduleProcHelper()

    b_scope = 'buffer0'

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph, dst_scope=b_scope)
    
    c_buf = tvm.compute(shape, lambda *i: a_buf(*i) + b_buf(*i), 'c_buf')
    sph.MarkScope(c_buf, 'buffer3')
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    mul_buf = tvm.compute(shape, 
                lambda *i: a_buf(*i).astype(dtype_w) * b_buf(*i).astype(dtype_w), 'mul_buf')
    sph.MarkScope(mul_buf)
    mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    gtm_buf = tvm.compute(shape, 
                lambda *i: tvm.max(a_buf(*i), b_buf(*i)), 'gtm_buf')
    sph.MarkScope(gtm_buf)
    gtm_host, gtm_dram = nnpu.utils.CopyBufToH(gtm_buf, 'gtm', sph)

    s = tvm.create_schedule([c_host.op, mul_host.op, gtm_host.op])
    sph.Transform(s)
    
    # x = s[c_buf].fuse(*c_buf.op.axis)
    # xo, xi = s[c_buf].split(x, factor=nvctr_unit)
    params = dict()
    params['code'] = 'binary'
    params['size'] = 64
    xi = s[c_buf].op.axis[1]
    s[c_buf].pragma(xi, 'nnpu.vector', str(params))

    x = s[mul_buf].fuse(*mul_buf.op.axis)
    xo, xi = s[mul_buf].split(x, factor=nvctr_unit)
    s[mul_buf].pragma(xi, 'nnpu.vector', str(params))
    
    x = s[gtm_buf].fuse(*gtm_buf.op.axis)
    xo, xi = s[gtm_buf].split(x, factor=nvctr_unit)
    s[gtm_buf].pragma(xi, 'nnpu.vector', str(params))

    print(tvm.lower(s, [a, b, c_host, mul_host, gtm_host], simple_mode=True))
    print(nnpu.lower(s, [a, b, c_host, mul_host, gtm_host], simple_mode=True))
    # exit()
    func = nnpu.build(s, [a, b, c_host, mul_host, gtm_host], 'nnpu', 'llvm', name='nnpu_exp')
    print('------------------- device module 1 IR: ')
    print(func.imported_modules[0].get_source('ir'))

    print('------------------- device module 1 micro code: ')
    print(func.imported_modules[0].get_source('uop'))

    print(s[c_buf].op.axis[0].dom)
    # exit()

    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=shape, dtype=a.dtype, low = -64, high = 63)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=shape, dtype=b.dtype, low = -64, high = 63)    
    b_nd = tvm.nd.array(b_np, ctx)
    
    c_nd = tvm.nd.array(np.zeros(shape).astype(c_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros(shape).astype(mul_host.dtype), ctx)
    gtm_nd = tvm.nd.array(np.zeros(shape).astype(gtm_host.dtype), ctx)

    # print('------------------- device module 1 llvm IR: ')
    # print(func.imported_modules[0].get_source('ll'))

    # print('------------------- device module 1 asm code: ')
    # print(func.imported_modules[0].get_source('asm'))

    func(a_nd, b_nd, c_nd, mul_nd, gtm_nd)
    # exit()
    # print('a = ')
    # print(a_np)
    # print('b = ')
    # print(b_np)
    # print('a + b =')
    # print(c_nd.asnumpy())
    # print("numpy ground truth is")
    gt = a_np + b_np
    # print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)
    # print('(int16)a * b =')
    # print(mul_nd.asnumpy())
    np.testing.assert_allclose(mul_nd.asnumpy(), np.multiply(a_np, b_np, dtype=mul_host.dtype))
    # print('max(a, b) = ')
    # print(gtm_nd.asnumpy())
    gt = np.maximum(a_np, b_np)
    np.testing.assert_allclose(gtm_nd.asnumpy(), gt)
    print('test passed!!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test of NNPU Op')
    parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
    args = parser.parse_args()

    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    test()