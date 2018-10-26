import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np


def test():
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (16,)
    bigshape =(4,64)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    sph = ScheduleProcHelper()

    a = tvm.placeholder(bigshape, dtype_n, 'a')
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)

    str_op = 'VAddMerge'
    k = tvm.reduce_axis((0, 4), 'k')
    c_buf = tvm.compute((64,), lambda i: tvm.sum(a_buf[k,i], axis=k), 'c_buf')
    
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c',sph)

    s = tvm.create_schedule(c_host.op)
    sph.Transform(s)
    #tensorize
    ko, ki = s[c_buf].split(c_buf.op.reduce_axis[0], factor=1)
    xo,xi = s[c_buf].split(c_buf.op.axis[0], factor=shape[0])
    s[c_buf].reorder(xo, ko, ki, xi)
    s[c_buf].tensorize(ki, env.intrins.get(str_op,  mode='n'))
    
    print(nnpu.lower(s, [a, c_host], simple_mode=True))
    #exit()
    func = nnpu.build(s, [a, c_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=bigshape, dtype=a.dtype, low = -4, high = 4)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros((64,), dtype=c_host.dtype), ctx)

    func(a_nd, c_nd)
    print(str_op)
    print(c_nd.asnumpy())
    gt = np.sum(a_np, axis=0, dtype=dtype_w)
    print('ground truth=')
    print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)

if __name__ == '__main__':
    test()