import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (2 , 2, 16)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder(shape, dtype_w, 'a')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)

    k = tvm.reduce_axis((0, 2), 'k')
    add_buf = tvm.compute((2, 16), lambda i,j: tvm.sum(a_buf[k, i , j], axis=k), 'add_buf')
    sph.MarkScope(add_buf)
    add_host, add_dram = nnpu.utils.CopyBufToH(add_buf, 'add', sph)

    k1 = tvm.reduce_axis((0, 2), 'k1')
    mul_buf = tvm.compute((2, 16), lambda i,j: tvm.sum(a_buf[k1, i,j], axis=k1), 'mul_buf')
    sph.MarkScope(mul_buf)
    mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    s = tvm.create_schedule([add_host.op,mul_host.op])
    sph.Transform(s)

    ko, ki = s[add_buf].split(add_buf.op.reduce_axis[0], factor=1)
    s[add_buf].reorder(ko, ki, *(s[add_buf].op.axis))
    s[add_buf].tensorize(ki, env.intrins.get('MAddMerge',shape=shape, mode='w'))

    ko1, ki1 = s[mul_buf].split(mul_buf.op.reduce_axis[0], factor=1)
    s[mul_buf].reorder(ko1, ki1, *(s[mul_buf].op.axis))
    s[mul_buf].tensorize(ki1, env.intrins.get('MMulMerge',shape=shape, mode='w'))


    print(nnpu.lower(s, [a, add_host,mul_host], simple_mode=True))

    func = nnpu.build(s, [a, add_host,mul_host], 'nnpu', 'llvm', name='nnpu_func')
    #exit()
    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(2,2, 16), dtype=a.dtype, low = -16, high = 16)
    a_nd = tvm.nd.array(a_np, ctx)
    
    add_nd = tvm.nd.array(np.zeros((2,16)).astype(add_host.dtype), ctx)

    mul_nd = tvm.nd.array(np.zeros((2,16)).astype(mul_host.dtype), ctx)
    
    func(a_nd, add_nd,mul_nd)

    print('a = ')
    print(a_np)
    print('reduce sum row = ')
    print(add_nd.asnumpy())
    print('ground truth is: ')
    gt = np.sum(a_np, axis=0)
    print(gt)
    np.testing.assert_allclose(add_nd.asnumpy(), gt)

    print('reduce mul row = ')
    print(mul_nd.asnumpy())
    gt = np.multiply.reduce(a_np ,axis=0,dtype = a.dtype)
    print(gt)
    np.testing.assert_allclose(mul_nd.asnumpy(), gt)


if __name__ == '__main__':
    test()