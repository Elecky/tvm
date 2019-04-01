import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    with ScheduleProcHelper():
        env = nnpu.get_env()
        shape = (16, 16)
        nrow = shape[0]
        
        dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
        a = tvm.placeholder(shape, dtype_n, 'a')
        a_buf, _ = nnpu.utils.CopyHtoBuf(a, 'a')

        k = tvm.reduce_axis((0, 16), 'k')
        b_buf = tvm.compute((nrow, ), lambda i: tvm.sum(a_buf[i, k].astype(dtype_w), k), 'b_buf')
        nnpu.utils.MarkScope(b_buf)
        b_host, _ = nnpu.utils.CopyBufToH(b_buf, 'b')

        s = nnpu.create_schedule(b_host.op)

        s[b_buf].tensorize(s[b_buf].op.axis[0], env.intrins.get('MReduceSumRow', mode='inc', shape=(16, 16)))

        print(nnpu.lower(s, [a, b_host], simple_mode=True))

        func = nnpu.build(s, [a, b_host], 'nnpu', 'llvm', name='nnpu_exp')

        print('------------------- device module 1 llvm IR: ')
        print(func.imported_modules[0].get_source('ll'))

        print('------------------- device module 1 asm code: ')
        print(func.imported_modules[0].get_source('asm'))

        ctx = tvm.nd.TVMContext(13, 0)
        a_np = np.random.randint(size=(16, 16), dtype=a.dtype, low = 0, high = 127)
        a_nd = tvm.nd.array(a_np, ctx)
        
        b_nd = tvm.nd.array(np.zeros((16,)).astype(b_host.dtype), ctx)

        func(a_nd, b_nd)

        print('a = ')
        print(a_np)
        print('reduce sum row = ')
        print(b_nd.asnumpy())

        print('ground truth is: ')
        gt = np.sum(a_np, axis=1)
        print(gt)
        np.testing.assert_allclose(b_nd.asnumpy(), gt)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test of NNPU Op')
    parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
    args = parser.parse_args()
    
    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    test()