import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

with ScheduleProcHelper():
    env = nnpu.get_env()
    nnpu.set_device(env, type='S0')
    shape = (48, 48)
    insn_shape = (16, 16)

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder(shape, dtype_n, 'a')
    b = tvm.placeholder(shape, dtype_n, 'b')
    
    sph = ScheduleProcHelper.current

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b', sph)

    k = tvm.reduce_axis((0, shape[1]), 'k')
    dot_shape = (shape[0], )
    dot_buf = tvm.compute(dot_shape, 
                lambda i: tvm.sum(a_buf[i, k].astype(dtype_w) * 
                                     b_buf[i, k].astype(dtype_w), k), 
                'dot_buf')
    sph.MarkScope(dot_buf, 'acc')
    
    res_buf = nnpu.utils.CopyAccToBuf(dot_buf, 'res')
    
    res_host, _ = nnpu.utils.CopyBufToH(res_buf, 'res')

    # tensorize
    s = nnpu.create_schedule(res_host.op)
    xo, ro, xi, ri = s[dot_buf].tile(dot_buf.op.axis[0], dot_buf.op.reduce_axis[0],
                                     insn_shape[0], insn_shape[1])
    s[dot_buf].tensorize(xi, env.intrins.get('MRowDot', shape=insn_shape, 
                                             mode='inc', scope_out='acc'))

    print(nnpu.lower(s, [a, b, res_host], simple_mode=True))
    
    func = nnpu.build(s, [a, b, res_host], 'nnpu', 'llvm', name='nnpu_func')

    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))
    
    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=shape, dtype=a.dtype, low = -32, high = 32)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=shape, dtype=b.dtype, low = -32, high = 32)    
    b_nd = tvm.nd.array(b_np, ctx)
    c_nd = tvm.nd.array(np.zeros((shape[0], )).astype(res_host.dtype), ctx)

    func(a_nd, b_nd, c_nd)
    #print('a = ')
    #print(a_np)
    #print('b = ')
    #print(b_np)

    print(c_nd.asnumpy())
    print('ground truth is')
    gt = np.multiply(a_np, b_np, dtype=res_host.dtype)
    gt = np.sum(gt, axis=1)
    print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)
    print('test passed')