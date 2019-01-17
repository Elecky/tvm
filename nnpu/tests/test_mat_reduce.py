import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    print('aaaa')
    env = nnpu.get_env()
    nnpu.set_device(env, type='S1')
    shape = (16, 16)
    a = tvm.placeholder(shape, env.cfg['dtype_n'], 'a')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    k = tvm.reduce_axis((0, 16), 'k')
    b_buf = tvm.compute((16, ), lambda i: tvm.sum(a_buf[i, k].astype(dtype_w), k), 'b_buf')
    sph.MarkScope(b_buf)
    b_host, b_dram = nnpu.utils.CopyBufToH(b_buf, 'b', sph)

    s = tvm.create_schedule(b_host.op)
    sph.Transform(s)
    s[b_buf].tensorize(s[b_buf].op.axis[0], env.intrins.get('MReduceSumRow', shape=(16,16), mode='inc'))

    print(nnpu.lower(s, [a, b_host], simple_mode=True))

    func = nnpu.build(s, [a, b_host], 'nnpu', 'llvm', name='nnpu_exp')

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
    test()