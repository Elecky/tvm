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
    
    dtype_w = env.cfg['dtype_w']

    k = tvm.reduce_axis((0, 16), 'k')
    dot_buf = tvm.compute((8, ), 
                lambda i: tvm.sum(a_buf[i, k].astype(dtype_w) * b_buf[i, k].astype(dtype_w), k), 'dot_buf')
    sph.MarkScope(dot_buf)
    dot_host, dot_dram = nnpu.utils.CopyBufToH(dot_buf, 'sum', sph)

    s = tvm.create_schedule(dot_host.op)
    sph.Transform(s)

    s[dot_buf].tensorize(s[dot_buf].op.axis[0], env.intrins.get('MRowDot', shape=shape, mode='inc'))

    print(nnpu.lower(s, [a,b, dot_host], simple_mode=True))
    func = nnpu.build(s, [a,b, dot_host], 'nnpu', 'llvm', name='nnpu_func')

    print('------------------- device module 1 llvm IR: ')
    print(func.imported_modules[0].get_source('ll'))

    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))
    
    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(8, 16), dtype=a.dtype, low = -32, high = 32)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=(8, 16), dtype=b.dtype, low = -32, high = 32)    
    b_nd = tvm.nd.array(b_np, ctx)
    c_nd = tvm.nd.array(np.zeros((8, )).astype(dot_host.dtype), ctx)

    func(a_nd, b_nd, c_nd)
    #print('a = ')
    #print(a_np)
    #print('b = ')
    #print(b_np)

    print(c_nd.asnumpy())
    print('ground truth is')
    gt = np.multiply(a_np, b_np, dtype=dot_host.dtype)
    gt = np.sum(gt, axis=1)
    print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test of NNPU Op')
    parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
    args = parser.parse_args()

    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    test()