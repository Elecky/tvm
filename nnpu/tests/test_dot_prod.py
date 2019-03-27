import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env, type='S0')

    a = tvm.placeholder((4, 16), 'int16', 'a')
    b = tvm.placeholder((16, ), 'int16', 'b')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)
    k = tvm.reduce_axis((0, 16), 'k')
    c_buf = tvm.compute((4, 1), lambda i, j: tvm.sum(a_buf[i,k] * b_buf[k], axis=k), 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    s = tvm.create_schedule(c_host.op)
    sph.Transform(s)
    print(s[c_buf])
    s[c_buf].tensorize(s[c_buf].op.axis[1], env.intrins.get('VDotV', mode='w'))

    print(nnpu.lower(s, [a, b, c_host], simple_mode=True))
    func = nnpu.build(s, [a, b, c_host], 'nnpu', 'llvm', name='nnpu_func')
    
    print('------------------- device module 1 llvm IR: ')
    print(func.imported_modules[0].get_source('ll'))

    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))

    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(4, 16), dtype=a.dtype, low = 0, high = 64)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=(16, ), dtype=b.dtype, low = 0, high = 64)    
    b_nd = tvm.nd.array(b_np, ctx)
    c_nd = tvm.nd.array(np.zeros((4, 1)).astype(c_host.dtype), ctx)

    func(a_nd, b_nd, c_nd)
    print(c_nd.asnumpy())
    print("numpy ground truth is")
    print(np.dot(a_np, b_np))

if __name__ == '__main__':
    test()