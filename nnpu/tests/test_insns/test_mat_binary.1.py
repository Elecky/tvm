'''
this test is intented to test the tensorize pattern matcher of tvm.
for example, whether it can treat tensor of shape (256, ) as region (16, 16),
this is reasonable if hardware backend has Matrix Add instruction, but we want to use it to add a long vector.
if tensorize pattern matcher can do this, then there will be not need to reshape input tensor.
'''
import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    shape = (16, 16)
    flatten_shape = (shape[0] * shape[1],)
    a = tvm.placeholder(flatten_shape, env.cfg['dtype_n'], 'a')
    b = tvm.placeholder(flatten_shape, env.cfg['dtype_n'], 'b')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)
    
    sum_buf = tvm.compute(flatten_shape, lambda i: a_buf[i] + b_buf[i], 'sum_buf')
    sph.MarkScope(sum_buf)
    sum_host, sum_dram = nnpu.utils.CopyBufToH(sum_buf, 'sum', sph)

    s = tvm.create_schedule([sum_host.op])
    sph.Transform(s)

    xo, xi = s[sum_buf].split(sum_buf.op.axis[0], 16)
    s[sum_buf].tensorize(xo, env.intrins.get('MAddM', shape=shape, mode='n'))

    print(nnpu.lower(s, [a, b, sum_host], simple_mode=True))
    func = nnpu.build(s, [a, b, sum_host], 'nnpu', 'llvm', name='nnpu_exp')

    print('------------------- device module 1 llvm IR: ')
    print(func.imported_modules[0].get_source('ll'))

    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))

    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=flatten_shape, dtype=a.dtype, low = 0, high = 23)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=flatten_shape, dtype=b.dtype, low = 0, high = 23)    
    b_nd = tvm.nd.array(b_np, ctx)
    c_nd = tvm.nd.array(np.zeros(flatten_shape).astype(sum_host.dtype), ctx)

    func(a_nd, b_nd, c_nd)
    print('a = ')
    print(a_np)
    print('b = ')
    print(b_np)
    print('a + b = ')
    print(c_nd.asnumpy())
    print("numpy ground truth is")
    print(a_np + b_np)
    np.testing.assert_allclose(c_nd.asnumpy(), a_np + b_np)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test of NNPU Op')
    parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
    args = parser.parse_args()

    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    test()