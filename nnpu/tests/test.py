import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)

    a = tvm.placeholder((4, 4, 16), 'int16', 'a')
    #b = tvm.placeholder((16, ), 'int16', 'b')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    #b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)
    
    k = tvm.reduce_axis((0, 4), 'k0')
    c_buf = tvm.compute((4, 16), lambda i, j: tvm.sum(a_buf[k, i, j], axis=k), 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    s = tvm.create_schedule(c_host.op)
    sph.Transform(s)
    ko, ki = s[c_buf].split(c_buf.op.reduce_axis[0], factor=1)
    s[c_buf].reorder(c_buf.op.axis[0], ko, ki, c_buf.op.axis[1])
    s[c_buf].tensorize(ki, env.intrins.get('VAddMerge', mode='w', nDim=3))

    print(nnpu.lower(s, [a, c_host], simple_mode=True))
    func = nnpu.build(s, [a, c_host], 'nnpu', 'llvm', name='nnpu_exp')


    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(4, 4, 16), dtype=a.dtype, low = -4000, high = 4000)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros((4, 16)).astype(c_host.dtype), ctx)

    func(a_nd, c_nd)
    print(c_nd.asnumpy())
    print("numpy ground truth is")
    gt = np.sum(a_np, axis=0)
    print(gt)

if __name__ == '__main__':
    test()