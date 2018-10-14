import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)
    a = tvm.placeholder((16, ), 'int16', 'a')
    sph = ScheduleProcHelper()
    Imm = tvm.const(5, 'int16')
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    #c_buf = tvm.compute((16, ), lambda i: tvm.select(a_buf[i]>Imm,a_buf[i],Imm), 'c_buf')
    c_buf = tvm.compute((16, ), lambda i: Imm+a_buf[i], 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)
    s = tvm.create_schedule(c_host.op)
    sph.Transform(s)
    s[c_buf].tensorize(s[c_buf].op.axis[0], env.intrins.get('VAddI', imm_value=Imm.value,mode='w'))

    print(nnpu.lower(s, [a,c_host], simple_mode=True))
    func = nnpu.build(s, [a,c_host], 'nnpu', 'llvm', name='nnpu_vmuli')


    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(16, ), dtype=a.dtype, low = 0, high = 10)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)

    func(a_nd, c_nd)
    print(a_nd.asnumpy())
    print(c_nd.asnumpy())
    

if __name__ == '__main__':
    test()