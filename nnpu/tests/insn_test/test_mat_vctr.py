import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    shape = (4, 16)
    a = tvm.placeholder(shape, dtype_n, 'a')
    b = tvm.placeholder((16, ), dtype_n, 'b')
    
    sph = ScheduleProcHelper()

    a_buf, _ = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, _ = nnpu.utils.CopyHtoBuf(b, 'b', sph)

    sum_buf = tvm.compute(shape, lambda i, j: a_buf[i, j] + b_buf[j], 'sum_buf')
    sph.MarkScope(sum_buf)
    sum_host, _ = nnpu.utils.CopyBufToH(sum_buf, 'sum', sph)

    sub_buf = tvm.compute(shape, lambda i, j: a_buf[i, j] - b_buf[j], 'sub_buf')
    sph.MarkScope(sub_buf)
    sub_host, _ = nnpu.utils.CopyBufToH(sub_buf, 'sub', sph)

    mul_buf = tvm.compute(shape, 
        lambda i, j: a_buf[i, j].astype(dtype_w) * b_buf[j].astype(dtype_w), 'sub_buf')
    sph.MarkScope(mul_buf)
    mul_host, _ = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    s = tvm.create_schedule([sum_host.op, sub_host.op, mul_host.op])
    sph.Transform(s)
    s[sum_buf].pragma(sum_buf.op.axis[0], 'nnpu.vector', str({'code': 'matrix-vector', 'shape': shape}))
    s[sub_buf].pragma(sub_buf.op.axis[0], 'nnpu.vector', str({'code': 'matrix-vector', 'shape': shape}))
    s[mul_buf].pragma(mul_buf.op.axis[0], 'nnpu.vector', str({'code': 'matrix-vector', 'shape': shape}))

    print(nnpu.lower(s, [a, b, sum_host, sub_host, mul_host], simple_mode=True))
    func = nnpu.build(s, [a, b, sum_host, sub_host, mul_host], 'nnpu', 'llvm', name='nnpu_func')

    print('------------------- device module 1 llvm IR: ')
    print(func.imported_modules[0].get_source('ir'))

    print('------------------- device module 1 uop code: ')
    print(func.imported_modules[0].get_source('uop'))

    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(4, 16), dtype=a.dtype, low = 0, high = 64)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=(16, ), dtype=b.dtype, low = 0, high = 64)    
    b_nd = tvm.nd.array(b_np, ctx)
    sum_nd = tvm.nd.array(np.zeros(shape).astype(sum_host.dtype), ctx)
    sub_nd = tvm.nd.array(np.zeros(shape).astype(sub_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros(shape).astype(mul_host.dtype), ctx)

    func(a_nd, b_nd, sum_nd, sub_nd, mul_nd)
    gt = a_np + b_np
    np.testing.assert_allclose(sum_nd.asnumpy(), gt)

    gt = a_np - b_np
    np.testing.assert_allclose(sub_nd.asnumpy(), gt)

    gt = a_np.astype(dtype_w) * b_np
    np.testing.assert_allclose(mul_nd.asnumpy(), gt)
    print('test passed')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test of NNPU Op')
    parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
    args = parser.parse_args()
    with nnpu.Environment('./nnpu_config.yaml'):
        env = nnpu.get_env()
        nnpu.set_device(env, type=args.sim)

        test()