import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    with ScheduleProcHelper():
        env = nnpu.get_env()
        nnpu.set_device(env, type="S1")
        # nnpu.set_dump(True)

        dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
        shape = (48, )
        nvctr_unit = env.cfg['vector_unit']['size']
        assert shape[0] % nvctr_unit == 0, 'error'

        a = tvm.placeholder(shape, dtype_n, 'a')
        b = tvm.placeholder(shape, dtype_n, 'b')
        
        sph = ScheduleProcHelper.current

        a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
        b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)
        
        c_buf = tvm.compute(shape, lambda i: a_buf[i] + b_buf[i], 'c_buf')
        sph.MarkScope(c_buf)
        c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

        plus2 = tvm.compute(shape, lambda i: c_host[i] + tvm.const(2, 'int8'), 'plus2')

        s = tvm.create_schedule([plus2.op])
        sph.Transform(s)
        
        xo, xi = s[c_buf].split(c_buf.op.axis[0], factor=nvctr_unit)
        s[c_buf].tensorize(xi, env.intrins.get('VAddV', mode='n'))

        print(nnpu.lower(s, [a, b, c_host, plus2], simple_mode=True))
        # exit()
        func = nnpu.build(s, [a, b, c_host, plus2], 'nnpu', 'llvm', name='nnpu_exp')
        # exit()

        ctx = tvm.nd.TVMContext(13, 0)

        a_np = np.random.randint(size=shape, dtype=a.dtype, low = -64, high = 63)
        #a_np = np.random.random(size=shape).astype(a_host.dtype)
        a_nd = tvm.nd.array(a_np, ctx)

        b_np = np.random.randint(size=shape, dtype=b.dtype, low = -64, high = 63)    
        b_nd = tvm.nd.array(b_np, ctx)
        
        c_nd = tvm.nd.array(np.zeros(shape).astype(c_host.dtype), ctx)
        plus2_nd = tvm.nd.array(np.zeros(shape).astype(plus2.dtype), ctx)

        print('------------------- device module 1 llvm IR: ')
        print(func.imported_modules[0].get_source('ll'))

        print('------------------- device module 1 asm code: ')
        print(func.imported_modules[0].get_source('asm'))

        # exit()
        func(a_nd, b_nd, c_nd, plus2_nd)
        
        print('a = ')
        print(a_np)
        print('b = ')
        print(b_np)
        print('a + b =')
        print(c_nd.asnumpy())
        print("numpy ground truth is")
        gt = a_np + b_np
        print(gt)
        np.testing.assert_allclose(c_nd.asnumpy(), gt)
        print('test passed!!')


if __name__ == '__main__':
    test()