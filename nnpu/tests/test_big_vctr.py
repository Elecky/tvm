import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np


def test():
    env = nnpu.get_env()
    nnpu.set_device(env, type='S0')
    shape = (16,)
    bigshape =(128,)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    assert bigshape[0] % shape[0] == 0, 'the big vctr size is wrong' 

    n_sheet=bigshape[0]//shape[0]
    sph = ScheduleProcHelper()

    a = tvm.placeholder(bigshape, dtype_n, 'a')
    b = tvm.placeholder(bigshape, dtype_n, 'b')
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)

    strop = 'VAddV'

    c_buf = tvm.compute(bigshape, lambda *i: a_buf(*i) + b_buf(*i) , 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'sum',sph)

    s = tvm.create_schedule(c_host.op)
    sph.Transform(s)
    #tensorize
    xo,xi = s[c_buf].split(c_buf.op.axis[0], factor=shape[0])
    s[c_buf].reorder(xo,xi)
    s[c_buf].tensorize(xi, env.intrins.get(strop,  mode='n'))
    
    print(nnpu.lower(s, [a, b, c_host], simple_mode=True))

    func = nnpu.build(s, [a, b, c_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=bigshape, dtype=a.dtype, low = -4, high = 4)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=bigshape, dtype=b.dtype, low = -4, high = 4)
    b_nd = tvm.nd.array(b_np, ctx)

    c_nd = tvm.nd.array(np.zeros(bigshape, dtype=c_host.dtype), ctx)

    func(a_nd, b_nd, c_nd)
    print(strop)
    print(c_nd.asnumpy())
    gt = a_np + b_np
    np.testing.assert_allclose(c_nd.asnumpy(), gt)


if __name__ == '__main__':
    test()