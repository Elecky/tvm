import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (4, 16)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder(shape, dtype_w, 'a')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)

    k = tvm.reduce_axis((0, 4), 'k')
    b_buf = tvm.compute((16, ), lambda i: tvm.sum(a_buf[k, i], axis=k), 'b_buf')
    sph.MarkScope(b_buf)
    b_host, b_dram = nnpu.utils.CopyBufToH(b_buf, 'b', sph)

    s = tvm.create_schedule(b_host.op)
    sph.Transform(s)
    ko, ki = s[b_buf].split(b_buf.op.reduce_axis[0], factor=1)
    s[b_buf].reorder(ko, ki, s[b_buf].op.axis[0])
    s[b_buf].tensorize(ki, env.intrins.get('VAddMerge', mode='w'))

    print(nnpu.lower(s, [a, b_host], simple_mode=True))

    func = nnpu.build(s, [a, b_host], 'nnpu', 'llvm', name='nnpu_func')
    #exit()
    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(4, 16), dtype=a.dtype, low = -127, high = 127)
    a_nd = tvm.nd.array(a_np, ctx)
    
    b_nd = tvm.nd.array(np.zeros((16,)).astype(b_host.dtype), ctx)

    func(a_nd, b_nd)

    print('a = ')
    print(a_np)
    print('reduce sum row = ')
    print(b_nd.asnumpy())

    print('ground truth is: ')
    gt = np.sum(a_np, axis=0)
    print(gt)
    np.testing.assert_allclose(b_nd.asnumpy(), gt)

if __name__ == '__main__':
    test()