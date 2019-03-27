import nnpu
import tvm
import numpy as np
from nnpu.utils import ScheduleProcHelper
import nnpu.utils as utils
import argparse

def test():
    pass
    if (False):
        print('-----')
    with ScheduleProcHelper():
        env = nnpu.get_env()

        shape = (16, 64)
        a_host = tvm.placeholder(shape, env.cfg['dtype_n'], 'a_host')
        a_buf, _ = nnpu.utils.CopyHtoBuf(a_host, 'a')
        
        vctr_shape = (64, )
        b_host = tvm.placeholder(vctr_shape, env.cfg['dtype_n'], 'b_host')
        b_buf, _ = nnpu.utils.CopyHtoBuf(b_host, 'b')

        dtype_w = env.cfg['dtype_w']
        
        out_shape = (4, 16)
        k = tvm.reduce_axis((0, 16), 'k')
        c_buf = tvm.compute(out_shape, 
                        lambda j, i: 
                            tvm.sum(a_buf[i, j * 16 + k].astype(dtype_w) * 
                                    b_buf[j * 16 + k].astype(dtype_w), 
                                    axis=k))
        utils.MarkScope(c_buf)
        c_host, _ = utils.CopyBufToH(c_buf, 'c')

        s = nnpu.create_schedule(c_host.op)

        # mark variable scopes

        # tensorize
        s[c_buf].tensorize(s[c_buf].op.axis[1], env.intrins.get('GEMM', shape=(16, 16, 1), 
                            mode='inc', reduce=True))

        # build
        print(tvm.lower(s, [a_host, b_host, c_host], simple_mode=True))

        print(nnpu.lower(s, [a_host, b_host, c_host], simple_mode=True))
        #exit()
        func = nnpu.build(s, [a_host, b_host, c_host], 'nnpu', 'llvm', name='nnpu_exp')

        print('function built: ')
        print('------------------- device module 1 asm code: ')
        print(func.imported_modules[0].get_source('asm'))
        #print(func.get_source())

        # prepare data
        ctx = tvm.nd.TVMContext(13, 0)

        a_np = np.random.randint(size=shape, dtype=a_host.dtype, low = -32, high = 32)
        # a_np = np.ones(shape).astype(a_host.dtype)
        a_nd = tvm.nd.array(a_np, ctx)

        b_np = np.random.randint(size=vctr_shape, dtype=b_host.dtype, low = -16, high = 16)
        # b_np = np.ones(vctr_shape).astype(b_host.dtype)
        b_nd = tvm.nd.array(b_np, ctx)

        out_nd = tvm.nd.array(np.zeros(out_shape).astype(c_host.dtype), ctx)

        # run
        func(a_nd, b_nd, out_nd)

        print('run finished')

        print('a=')
        print(a_np)
        print('b=')
        print(b_np)
        print('out=')
        out_np = out_nd.asnumpy()
        out_np = np.sum(out_np, axis=0)
        print(out_np)

        print('numpy ground truth is: ')
        gt = np.dot(a_np.astype(dtype_w), b_np.astype(dtype_w))
        #gt = np.greater(np.dot(a_np.astype(dtype_w), b_np.astype(dtype_w)), bias_np)
        print(gt)

        np.testing.assert_allclose(out_np, gt)

def test2():
    with ScheduleProcHelper():
        env = nnpu.get_env()

        shape = (4, 16, 16)
        a_host = tvm.placeholder(shape, env.cfg['dtype_n'], 'a_host')
        a_buf, _ = nnpu.utils.CopyHtoBuf(a_host, 'a')
        
        vctr_shape = (64, )
        b_host = tvm.placeholder(vctr_shape, env.cfg['dtype_n'], 'b_host')
        b_buf, _ = nnpu.utils.CopyHtoBuf(b_host, 'b')

        dtype_w = env.cfg['dtype_w']
        
        out_shape = (4, 16)
        k = tvm.reduce_axis((0, 16), 'k')
        c_buf = tvm.compute(out_shape, 
                        lambda j, i: 
                            tvm.sum(a_buf[j, i, k].astype(dtype_w) * 
                                    b_buf[j * 16 + k].astype(dtype_w), 
                                    axis=k))
        utils.MarkScope(c_buf)
        c_host, _ = utils.CopyBufToH(c_buf, 'c')

        s = nnpu.create_schedule(c_host.op)

        # mark variable scopes

        # tensorize
        s[c_buf].tensorize(s[c_buf].op.axis[1], env.intrins.get('GEMM', shape=(16, 16, 1), 
                            mode='inc', reduce=True))

        # build
        print(tvm.lower(s, [a_host, b_host, c_host], simple_mode=True))

        print(nnpu.lower(s, [a_host, b_host, c_host], simple_mode=True))
        #exit()
        func = nnpu.build(s, [a_host, b_host, c_host], 'nnpu', 'llvm', name='nnpu_exp')

        print('function built: ')
        print('------------------- device module 1 asm code: ')
        print(func.imported_modules[0].get_source('asm'))
        #print(func.get_source())

        # prepare data
        ctx = tvm.nd.TVMContext(13, 0)

        a_np = np.random.randint(size=shape, dtype=a_host.dtype, low = -32, high = 32)
        # a_np = np.ones(shape).astype(a_host.dtype)
        a_nd = tvm.nd.array(a_np, ctx)

        b_np = np.random.randint(size=vctr_shape, dtype=b_host.dtype, low = -16, high = 16)
        # b_np = np.ones(vctr_shape).astype(b_host.dtype)
        b_nd = tvm.nd.array(b_np, ctx)

        out_nd = tvm.nd.array(np.zeros(out_shape).astype(c_host.dtype), ctx)

        # run
        func(a_nd, b_nd, out_nd)

        print('run finished')

        print('a=')
        print('[a is omitted to save space]')
        # print(a_np)
        print('b=')
        print(b_np)
        print('out=')
        out_np = out_nd.asnumpy()
        out_np = np.sum(out_np, axis=0)
        print(out_np)

        print('numpy ground truth is: ')
        a_np = np.transpose(a_np, axes=(1, 0, 2))
        a_np = np.reshape(a_np, newshape=(16, 64))
        gt = np.dot(a_np.astype(dtype_w), b_np.astype(dtype_w))
        #gt = np.greater(np.dot(a_np.astype(dtype_w), b_np.astype(dtype_w)), bias_np)
        print(gt)

        np.testing.assert_allclose(out_np, gt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test gemm with tiled/non-tiled data')
    parser.add_argument('--tiled', dest='tiled', action='store_const',
                        const=True, default=False)
    args = parser.parse_args()
    
    env = nnpu.get_env()
    nnpu.set_device(env, type='SC')

    if (args.tiled):
        test2()
    else:
        test()