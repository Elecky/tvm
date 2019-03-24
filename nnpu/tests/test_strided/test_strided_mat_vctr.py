import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env, type='S0')

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    shape = (32, 32)
    a = tvm.placeholder(shape, dtype_n, 'a')
    b = tvm.placeholder((shape[1], ), dtype_n, 'b')
    
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)

    sum_buf = tvm.compute(shape, lambda i, j: a_buf[i, j] + b_buf[j], 'sum_buf')
    sph.MarkScope(sum_buf)
    sum_host, sum_dram = nnpu.utils.CopyBufToH(sum_buf, 'sum', sph)

    sub_buf = tvm.compute(shape, lambda i, j: a_buf[i, j] - b_buf[j], 'sub_buf')
    sph.MarkScope(sub_buf)
    sub_host, sub_dram = nnpu.utils.CopyBufToH(sub_buf, 'sub', sph)

    mul_buf = tvm.compute(shape, 
        lambda i, j: a_buf[i, j].astype(dtype_w) * b_buf[j].astype(dtype_w), 'sub_buf')
    sph.MarkScope(mul_buf)
    mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    s = tvm.create_schedule([sum_host.op, sub_host.op, mul_host.op])
    sph.Transform(s)
    # tensorize
    insn_shape = (16, 16)
    xo, yo, xi, yi = s[sum_buf].tile(sum_buf.op.axis[0], sum_buf.op.axis[1], 
                                     insn_shape[0], insn_shape[1])
    s[sum_buf].tensorize(xi, env.intrins.get('MAddV', shape=insn_shape, mode='n'))

    xo, yo, xi, yi = s[sub_buf].tile(sub_buf.op.axis[0], sub_buf.op.axis[1], 
                                     insn_shape[0], insn_shape[1])
    s[sub_buf].tensorize(xi, env.intrins.get('MSubV', shape=insn_shape, mode='n'))

    xo, yo, xi, yi = s[mul_buf].tile(mul_buf.op.axis[0], mul_buf.op.axis[1], 
                                     insn_shape[0], insn_shape[1])
    s[mul_buf].tensorize(xi, env.intrins.get('MMulV', shape=insn_shape, mode='inc'))

    print(nnpu.lower(s, [a, b, sum_host, sub_host, mul_host], simple_mode=True))
    
    func = nnpu.build(s, [a, b, sum_host, sub_host, mul_host], 'nnpu', 'llvm', name='nnpu_func')


    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=shape, dtype=a.dtype, low = 0, high = 64)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=(shape[1], ), dtype=b.dtype, low = 0, high = 64)    
    b_nd = tvm.nd.array(b_np, ctx)
    sum_nd = tvm.nd.array(np.zeros(shape).astype(sum_host.dtype), ctx)
    sub_nd = tvm.nd.array(np.zeros(shape).astype(sub_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros(shape).astype(mul_host.dtype), ctx)

    func(a_nd, b_nd, sum_nd, sub_nd, mul_nd)
    print('a = ')
    print(a_np)
    print('b = ')
    print(b_np)
    print('sum result is ')
    print(sum_nd.asnumpy())
    print("numpy ground truth is")
    gt = a_np + b_np
    print(gt)
    np.testing.assert_allclose(sum_nd.asnumpy(), gt)

    print('sub result is ')
    print(sub_nd.asnumpy())
    np.testing.assert_allclose(sub_nd.asnumpy(), a_np - b_np)

    print('mul result is ')
    print(mul_nd.asnumpy())
    np.testing.assert_allclose(mul_nd.asnumpy(), a_np.astype(dtype_w) * b_np)

if __name__ == '__main__':
    test()