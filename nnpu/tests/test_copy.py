import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='test of NNPU Op')
parser.add_argument('--sim', type=str, help='the simulator to use', 
                    default='S0', choices=['S0', 'S1', 'SC'])
args = parser.parse_args()

env = nnpu.get_env()
nnpu.set_device(env, type=args.sim)

with (ScheduleProcHelper()):
    env = nnpu.get_env()
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

    a = tvm.placeholder((32, 32), dtype_w, 'a')
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a')
    
    trans_buf = nnpu.utils.transpose(a_buf, (1, 0))

    re_buf = nnpu.utils.reshape(trans_buf, (2, 16, 2, 16))
    
    tile_buf = nnpu.utils.transpose(re_buf, (0, 2, 1, 3))
    tile_host, _ = nnpu.utils.CopyBufToH(tile_buf, 'tile')
    
    s = nnpu.create_schedule([tile_host.op])

    print(tvm.lower(s, [a, tile_host], simple_mode=True))
    print(nnpu.lower(s, [a, tile_host], simple_mode=True))
    func = nnpu.build(s, [a, tile_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(32, 32), dtype=a.dtype, low = -10000, high = 10000)
    a_nd = tvm.nd.array(a_np, ctx)

    #b_np = np.random.randint(size=(4, 32), dtype=b.dtype, low = -10000, high = 10000)
    #b_nd = tvm.nd.array(b_np, ctx)

    re_nd = tvm.nd.array(np.zeros((2, 2, 16, 16), dtype=tile_host.dtype), ctx)

    print('------------------- device module 1 llvm IR: ')
    print(func.imported_modules[0].get_source('ll'))

    print('------------------- device module 1 asm code: ')
    print(func.imported_modules[0].get_source('asm'))
    
    func(a_nd, re_nd)

    #print(a_nd)
    #print(re_nd.asnumpy())
    gt = np.transpose(a_np, (1, 0))
    gt = np.reshape(gt, (2, 16, 2, 16))
    gt = np.transpose(gt, (0, 2, 1, 3))
    np.testing.assert_allclose(re_nd.asnumpy(), gt)

    print('test passed')