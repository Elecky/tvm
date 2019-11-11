import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np
    
with (ScheduleProcHelper()):
    env = nnpu.get_env()
    nnpu.set_device(env, type='S0')
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    a = tvm.placeholder((2, 4, 16), dtype_n, 'a')
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a')
    
    pad_buf = tvm.compute((2, 6, 16), 
                        lambda i, j, k: tvm.expr.Select(j >= 2, a_buf[i, j - 2, k], tvm.const(0, dtype_n)),
                        'pad')
    nnpu.utils.MarkScope(pad_buf)
    nnpu.utils.PragmaCopy(pad_buf)
    tile_host, _ = nnpu.utils.CopyBufToH(pad_buf, 'tile')
    
    s = nnpu.create_schedule([tile_host.op])

    print(tvm.lower(s, [a, tile_host], simple_mode=True))
    print(nnpu.lower(s, [a, tile_host], simple_mode=True))
    # exit(0)
    func = nnpu.build(s, [a, tile_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(2, 4, 16), dtype=a.dtype, low = -128, high = 127)
    a_nd = tvm.nd.array(a_np, ctx)

    #b_np = np.random.randint(size=(4, 32), dtype=b.dtype, low = -10000, high = 10000)
    #b_nd = tvm.nd.array(b_np, ctx)

    re_nd = tvm.nd.array(np.zeros((2, 6, 16), dtype=tile_host.dtype), ctx)

    print('------------------- device module 1 llvm IR: ')
    print(func.imported_modules[0].get_source('ll'))

    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))
    
    func(a_nd, re_nd)

    print(a_nd)
    print(re_nd.asnumpy())

    prin#t('test passed')