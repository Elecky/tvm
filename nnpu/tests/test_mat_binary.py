import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    shape = (8, 16)
    a = tvm.placeholder(shape, env.cfg['dtype_n'], 'a')
    b = tvm.placeholder(shape, env.cfg['dtype_n'], 'b')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)
    
    sum_buf = tvm.compute(shape, lambda i, j: a_buf[i, j] + b_buf[i, j], 'sum_buf')
    sph.MarkScope(sum_buf)
    sum_host, sum_dram = nnpu.utils.CopyBufToH(sum_buf, 'sum', sph)

    sub_buf = tvm.compute(shape, lambda i, j: a_buf[i, j] - b_buf[i, j], 'sum_buf')
    sph.MarkScope(sub_buf)
    sub_host, sub_dram = nnpu.utils.CopyBufToH(sub_buf, 'sub', sph)

    dtype_w = env.cfg['dtype_w']
    mul_buf = tvm.compute(shape, 
                lambda i, j: a_buf[i, j].astype(dtype_w) * b_buf[i, j].astype(dtype_w), 'sum_buf')
    sph.MarkScope(mul_buf)
    mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    s = tvm.create_schedule([sum_host.op, sub_host.op, mul_host.op])
    sph.Transform(s)
    s[sum_buf].tensorize(s[sum_buf].op.axis[0], env.intrins.get('MAddM', shape=shape, mode='n'))
    s[sub_buf].tensorize(s[sub_buf].op.axis[0], env.intrins.get('MSubM', shape=shape, mode='n'))
    s[mul_buf].tensorize(s[mul_buf].op.axis[0], env.intrins.get('MMulM', shape=shape, mode='inc'))

    print(nnpu.lower(s, [a, b, sum_host, sub_host, mul_host], simple_mode=True))
    func = nnpu.build(s, [a, b, sum_host, sub_host, mul_host], 'nnpu', 'llvm', name='nnpu_exp')

    print('------------------- device module 1 llvm IR: ')
    print(func.imported_modules[0].get_source('ll'))

    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))

    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(8, 16), dtype=a.dtype, low = 0, high = 23)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=(8, 16), dtype=b.dtype, low = 0, high = 23)    
    b_nd = tvm.nd.array(b_np, ctx)
    c_nd = tvm.nd.array(np.zeros((8, 16)).astype(sum_host.dtype), ctx)

    sub_nd = tvm.nd.array(np.zeros((8, 16)).astype(sub_host.dtype), ctx)
    sub_nd = tvm.nd.array(np.zeros((8, 16)).astype(sub_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros((8, 16)).astype(mul_host.dtype), ctx)

    func(a_nd, b_nd, c_nd, sub_nd, mul_nd)
    print('a = ')
    print(a_np)
    print('b = ')
    print(b_np)
    print('a + b = ')
    print(c_nd.asnumpy())
    print("numpy ground truth is")
    print(a_np + b_np)
    np.testing.assert_allclose(c_nd.asnumpy(), a_np + b_np)
    print('a - b = ')
    print(sub_nd.asnumpy())
    np.testing.assert_allclose(sub_nd.asnumpy(), a_np - b_np)
    print('a HM b = ')
    print(mul_nd.asnumpy())
    np.testing.assert_allclose(mul_nd.asnumpy(), 
            np.multiply(a_np.astype(dtype_w), b_np.astype(dtype_w)))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test of NNPU Op')
    parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
    args = parser.parse_args()

    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    test()