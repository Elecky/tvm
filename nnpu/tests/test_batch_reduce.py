import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np
#add new func
def test():
    env = nnpu.get_env()
    nnpu.set_device(env, type='SC')
    shape = (4, 16)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder(shape, dtype_w, 'a')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)

    k = tvm.reduce_axis((0, 4), 'k')
    add_buf = tvm.compute((16, ), lambda i: tvm.sum(a_buf[k, i], axis=k), 'add_buf')
    sph.MarkScope(add_buf)
    add_host, add_dram = nnpu.utils.CopyBufToH(add_buf, 'add', sph)

    # k1 = tvm.reduce_axis((0, 4), 'k1')
    # mul_buf = tvm.compute((16, ), lambda i: tvm.sum(a_buf[k1, i], axis=k1), 'mul_buf')
    # sph.MarkScope(mul_buf)
    # mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    k2 = tvm.reduce_axis((0, 4), 'k2')
    gtm_buf = tvm.compute((16, ), lambda i: tvm.max(a_buf[k2, i], axis=k2), 'gtm_buf')
    sph.MarkScope(gtm_buf)
    gtm_host, gtm_dram = nnpu.utils.CopyBufToH(gtm_buf, 'gtm', sph)

    s = tvm.create_schedule([add_host.op, gtm_host.op])
    sph.Transform(s)

    ko, ki = s[add_buf].split(add_buf.op.reduce_axis[0], factor=1)
    s[add_buf].reorder(ko, ki, s[add_buf].op.axis[0])
    s[add_buf].tensorize(ki, env.intrins.get('VAddMerge', mode='w'))

    # ko1, ki1 = s[mul_buf].split(mul_buf.op.reduce_axis[0], factor=1)
    # s[mul_buf].reorder(ko1, ki1, s[mul_buf].op.axis[0])
    # s[mul_buf].tensorize(ki1, env.intrins.get('VMulMerge', mode='w'))

    ko2, ki2 = s[gtm_buf].split(gtm_buf.op.reduce_axis[0], factor=1)
    s[gtm_buf].reorder(ko2, ki2, s[gtm_buf].op.axis[0])
    s[gtm_buf].tensorize(ki2, env.intrins.get('VGTMMerge', mode='w'))

    print(nnpu.lower(s, [a, add_host,gtm_host], simple_mode=True))

    func = nnpu.build(s, [a, add_host,gtm_host], 'nnpu', 'llvm', name='nnpu_func')
    #exit()
    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(4, 16), dtype=a.dtype, low = -16, high = 16)
    a_nd = tvm.nd.array(a_np, ctx)
    
    add_nd = tvm.nd.array(np.zeros((16,)).astype(add_host.dtype), ctx)

    # mul_nd = tvm.nd.array(np.zeros((16,)).astype(mul_host.dtype), ctx)

    gtm_nd = tvm.nd.array(np.zeros((16,)).astype(gtm_host.dtype), ctx)
    
    func(a_nd, add_nd,gtm_nd)
    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))

    print('a = ')
    print(a_np)
    print('reduce sum row = ')
    print(add_nd.asnumpy())
    print('ground truth is: ')
    gt = np.sum(a_np, axis=0)
    print(gt)
    np.testing.assert_allclose(add_nd.asnumpy(), gt)

    # print('reduce mul row = ')
    # print(mul_nd.asnumpy())
    # gt = np.multiply.reduce(a_np ,axis=0,dtype = a.dtype)
    # print(gt)
    # np.testing.assert_allclose(mul_nd.asnumpy(), gt)

    print('reduce max row = ')
    print(gtm_nd.asnumpy())
    gt = np.max(a_np ,axis=0)
    print(gt)
    np.testing.assert_allclose(gtm_nd.asnumpy(), gt)

if __name__ == '__main__':
    test()