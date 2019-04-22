import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    shape = (256, )
    nvctr_unit = env.cfg['vector_unit']['size']
    assert shape[0] % nvctr_unit == 0, 'error'

    a = tvm.placeholder(shape, dtype_n, 'a')
    b = tvm.placeholder(shape, dtype_n, 'b')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)
    
    c_buf = tvm.compute(shape, lambda i: a_buf[i] + b_buf[i], 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    s = tvm.create_schedule(c_host.op)
    sph.Transform(s)

    xo, xi = s[c_buf].split(c_buf.op.axis[0], factor=nvctr_unit)
    s[c_buf].tensorize(xi, env.intrins.get('VAddV', mode='n'))

    # split and compute_at
    core_number = 4
    xo, xi = s[c_host].split(s[c_host].op.axis[0], nparts=core_number)
    s[c_host].pragma(xi, env.dma_copy_from_buf)

    s[c_buf].compute_at(s[c_host], xo)
    s[a_buf].compute_at(s[c_host], xo)
    s[b_buf].compute_at(s[c_host], xo)

    # bind axis
    s[c_host].bind(xo, tvm.thread_axis("coreIdx"))

    print(nnpu.lower(s, [a, b, c_host], simple_mode=True))
    # exit()
    func = nnpu.build(s, [a, b, c_host], 'nnpu', 'llvm', name='nnpu_func')
    # exit()

    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=shape, dtype=a.dtype, low = -64, high = 63)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=shape, dtype=b.dtype, low = -64, high = 63)    
    b_nd = tvm.nd.array(b_np, ctx)
    
    c_nd = tvm.nd.array(np.zeros(shape).astype(c_host.dtype), ctx)

    print('------------------- device module 1 llvm IR: ')
    print(func.imported_modules[0].get_source('ll'))

    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))

    func(a_nd, b_nd, c_nd)
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