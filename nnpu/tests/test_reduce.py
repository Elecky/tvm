import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)

    a = tvm.placeholder((4, 16), 'int16', 'a')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)

    k = tvm.reduce_axis((0, 16), 'k')
    c_buf = tvm.compute((4, 1), lambda i, j: tvm.sum(a_buf[i,k], axis=k), 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    k1 = tvm.reduce_axis((0, 16), 'k1')
    max_buf = tvm.compute((4, 1), lambda i, j: tvm.max(a_buf[i,k1], axis=k1), 'max_buf')
    sph.MarkScope(max_buf)
    max_host, max_dram = nnpu.utils.CopyBufToH(max_buf, 'max', sph)

    k2 = tvm.reduce_axis((0, 16), 'k2')
    min_buf = tvm.compute((4, 1), lambda i, j: tvm.min(a_buf[i,k2], axis=k2), 'min_buf')
    sph.MarkScope(min_buf)
    min_host, min_dram = nnpu.utils.CopyBufToH(min_buf, 'min', sph)

    # create schedule and tensorize
    s = tvm.create_schedule([c_host.op, max_host.op, min_host.op])
    sph.Transform(s)
    s[c_buf].tensorize(s[c_buf].op.axis[1], env.intrins.get('VReduceSum', mode='w'))
    s[max_buf].tensorize(s[max_buf].op.axis[1], env.intrins.get('VReduceMax', mode='w'))
    s[min_buf].tensorize(s[min_buf].op.axis[1], env.intrins.get('VReduceMin', mode='w'))

    # build
    print(nnpu.lower(s, [a, c_host, max_host, min_host], simple_mode=True))
    func = nnpu.build(s, [a, c_host, max_host, min_host], 'nnpu', 'llvm', name='nnpu_func')

    # create data and run

    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(4, 16), dtype=a.dtype, low = 0, high = 64)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros((4, 1)).astype(c_host.dtype), ctx)
    max_nd = tvm.nd.array(np.zeros((4, 1)).astype(c_host.dtype), ctx)
    min_nd = tvm.nd.array(np.zeros((4, 1)).astype(c_host.dtype), ctx)

    func(a_nd, c_nd, max_nd, min_nd)

    # check results
    print('a = ')
    print(a_np)
    print('reduce sum result is:')
    print(c_nd.asnumpy())
    print("numpy ground truth is")
    gt = np.sum(a_np, axis=(1,), keepdims=True)
    print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)

    print('reduce max result is:')
    print(max_nd.asnumpy())
    np.testing.assert_allclose(max_nd.asnumpy(), np.max(a_np, axis=(1,), keepdims=True))

    print('reduce min result is:')
    print(min_nd.asnumpy())
    np.testing.assert_allclose(min_nd.asnumpy(), np.min(a_np, axis=(1,), keepdims=True))

if __name__ == '__main__':
    test()