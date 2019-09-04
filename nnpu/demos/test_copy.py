'''
reshape and permute demo.
====================
reshape and premute are essentially memory copies, they are both implemented by Scratchpad Copy instruction.
in this demo, we try to do the following operation on NNPU:
    b = np.transpose(a, (1, 0))
    c = np.reshape(b, (2, 16, 2, 16))
    d = np.transpose(c, (0, 2, 1, 3))
'''
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

    a = tvm.placeholder((8, 8), dtype_w, 'a')
    
    #=================================================================#
    # ------ begin compute describing. ------
    #=================================================================#
    # copy to scratchpad.
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a')

    # here we simply use some helper function to do the reshape and transpose.
    trans_buf = nnpu.utils.transpose(a_buf, (1, 0))
    re_buf = nnpu.utils.reshape(trans_buf, (2, 4, 2, 4), dst_scope='buffer1')
    tile_buf = nnpu.utils.transpose(re_buf, (0, 2, 1, 3), dst_scope='buffer2')
    # copy back to host.
    tile_host, _ = nnpu.utils.CopyBufToH(tile_buf, 'tile')
    # ------ this ends the computation description. ------

    #==================================#
    # ------ begin scheduling ------
    #==================================#
    s = nnpu.create_schedule([tile_host.op])

    # since all operations are scratchpad copy, all we need to do is pragma.
    # this is done by the helper functions, so nothing to do here.
    
    #==================================#
    # ------ this ends the scheduling ------
    #==================================#

    print(tvm.lower(s, [a, tile_host], simple_mode=True))
    print(nnpu.lower(s, [a, tile_host], simple_mode=True))
    func = nnpu.build(s, [a, tile_host], 'nnpu', 'llvm', name='nnpu_func')
    print('------------------- device module 1 TVM IR: ')
    print(func.imported_modules[0].get_source('ir'))
    print('------------------- device module 1 uop: ')
    print(func.imported_modules[0].get_source('uop'))
    # exit()

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(8, 8), dtype=a.dtype, low = -10000, high = 10000)
    a_nd = tvm.nd.array(a_np, ctx)

    #b_np = np.random.randint(size=(4, 32), dtype=b.dtype, low = -10000, high = 10000)
    #b_nd = tvm.nd.array(b_np, ctx)

    re_nd = tvm.nd.array(np.zeros((2, 2, 4, 4), dtype=tile_host.dtype), ctx)
    
    func(a_nd, re_nd)

    #print(a_nd)
    # print(re_nd.asnumpy())
    gt = np.transpose(a_np, (1, 0))
    gt = np.reshape(gt, (2, 4, 2, 4))
    gt = np.transpose(gt, (0, 2, 1, 3))
    # print(gt)
    np.testing.assert_allclose(re_nd.asnumpy(), gt)

    print('test passed')