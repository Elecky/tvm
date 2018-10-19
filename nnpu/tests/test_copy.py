import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder((32,), dtype_w, 'a')
    b = tvm.placeholder((4, 32), dtype_w, 'b')
    
    sph = ScheduleProcHelper()

    #a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)

    #re_buf = tvm.compute((2, 16), lambda i, j: a_buf[i * 16 + j], 'reshaped')
    #sph.MarkScope(re_buf)
    #re_host, re_dram = nnpu.utils.CopyBufToH(re_buf, 'reshaped', sph)

    #trans_buf = tvm.compute((16, 4), lambda i, j: re_buf[j, i], 'transed')
    #sph.MarkScope(trans_buf)
    #trans_host, trans_dram = nnpu.utils.CopyBufToH(trans_buf, 'transed', sph)

    tiled_buf = tvm.compute((2, 2, 2, 16), 
                    lambda i0, j0, i1, j1: b_buf[i0 * 2 + i1, j0 * 16 + j1], 'tiled_buf')
    sph.MarkScope(tiled_buf)
    tiled_host, tiled_dram = nnpu.utils.CopyBufToH(tiled_buf, 'tiled', sph)

    #r = tvm.reduce_axis((0, 16), 'r')
    #mul_buf = tvm.compute((2, 2, 16),
    #                lambda i, j, k: tvm.sum(tiled_buf[i, j, k, r] * re_buf[j, r], r), 'mul')
    #sph.MarkScope(mul_buf)
    #mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)
#
    #out_buf = tvm.compute((2, 16),
    #                lambda i, j: mul_buf[i, 0, j] + mul_buf[i, 1, j], 'out')
    #sph.MarkScope(out_buf)
    #out_host, out_dram = nnpu.utils.CopyBufToH(out_buf, 'out', sph)

    s = tvm.create_schedule([tiled_host.op])
    sph.Transform(s)
    #s[re_buf].pragma(s[re_buf].op.axis[0], env.scratchpad_copy)
    s[tiled_buf].pragma(s[tiled_buf].op.axis[0], env.scratchpad_copy)
    #s[mul_buf].tensorize(s[mul_buf].op.axis[2], 
    #                     env.intrins.get('GEMM', shape=(16, 16, 1), mode='w', reduce=True))
    #s[out_buf].tensorize(s[out_buf].op.axis[1], env.intrins.get('VAddV', mode='w'))

    print(tvm.lower(s, [b, tiled_host], simple_mode=True))
    print(nnpu.lower(s, [b, tiled_host], simple_mode=True))
    func = nnpu.build(s, [b, tiled_host], 'nnpu', 'llvm', name='nnpu_exp')

    ctx = tvm.nd.TVMContext(13, 0)
    #a_np = np.random.randint(size=(64, ), dtype=a.dtype, low = -10000, high = 10000)

    b_np = np.random.randint(size=(4, 32), dtype=b.dtype, low = -10000, high = 10000)
    b_nd = tvm.nd.array(b_np, ctx)

    t_nd = tvm.nd.array(np.zeros((2, 2, 2, 16), dtype=tiled_host.dtype), ctx)

    func(b_nd, t_nd)

    print(b_np)
    print(t_nd.asnumpy())

if __name__ == '__main__':
    test()