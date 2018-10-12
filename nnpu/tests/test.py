import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)

    a = tvm.placeholder((16, ), 'int16', 'a')
    b = tvm.placeholder((16, ), 'int16', 'b')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)
    
    c_buf = tvm.compute((16, ), lambda i: a_buf[i] + b_buf[i], 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    s = tvm.create_schedule(c_host.op)
    sph.Transform(s)
    s[c_buf].tensorize(s[c_buf].op.axis[0], env.intrins.get('VAddV', mode='w'))

    print(nnpu.lower(s, [a, b, c_host], simple_mode=True))
    func = nnpu.build(s, [a, b, c_host], 'nnpu', 'llvm', name='nnpu_exp')


    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(16, ), dtype=a.dtype, low = 0, high = 10000)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=(16, ), dtype=b.dtype, low = 0, high = 10000)    
    b_nd = tvm.nd.array(b_np, ctx)
    c_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)

    func(a_nd, b_nd, c_nd)
    print(c_nd.asnumpy())
    print("numpy ground truth is")
    print(a_np + b_np)

if __name__ == '__main__':
    test()