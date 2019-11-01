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
nnpu.set_device(env, type='S0')

with ScheduleProcHelper():
    fh, fw, fc = 18, 18, 16
    feature = tvm.placeholder((fh, fw, fc), 'int8', 'feature')
    kh, kw = 3, 3
    ph, pw = fh - kh + 1, fw- kw + 1
    packed_shape = (ph * pw, kh * kw * fc)
    packed = tvm.compute(packed_shape, lambda p, c: feature[p / pw + c / fc / kw, p % pw + (c / fc) % kw, c % fc], 'packed')
    tiled = tvm.compute((packed_shape[0] // 8, packed_shape[1] // 8, 8, 8),
                        lambda po, co, pi, ci: packed[po * 8 + pi, co * 8 + ci],
                        'tiled')
    s = tvm.create_schedule(tiled.op)

    gt_func = tvm.build(s, [feature, tiled], 'llvm', 'llvm', 'gt_func')

    feature_buf = s.cache_read(feature, env.get_scope('buffer0'), packed)
    tiled_buf = s.cache_write(tiled, env.get_scope('buffer0'))

    s[packed].compute_inline()


    po, co, pi, ci = s[tiled_buf].op.axis
    if (fc >= 8):
        outter, co = s[tiled_buf].split(co, fc // 8)
    else:
        outter, co = s[tiled_buf].split(co, 1)
    c1, c2 = s[tiled_buf].split(outter, kw)
    ph, pwo = s[tiled_buf].split(po, pw // 8)
    s[tiled_buf].reorder(co, pwo, ph, pi, c1, c2, ci)
    s[tiled_buf].pragma(pwo, 'nnpu.im2col')
    s[feature_buf].pragma(s[feature_buf].leaf_iter_vars[0], env.dma_copy_to_buf)
    s[tiled].pragma(s[tiled].leaf_iter_vars[0], env.dma_copy_from_buf)
    # pw = s[tiled].fuse(pwo, pi)
    # s[tiled_buf].reorder(co, ph, pw, c1, c2, ci)
    
    print(nnpu.lower(s, [feature, tiled], simple_mode=True))
    func = nnpu.build(s, [feature, tiled], 'nnpu', 'llvm', 'im2col_func')
    print('------------------- device module 1 TVM IR: ')
    print(func.imported_modules[0].get_source('ir'))
    print('------------------- device module 1 uop: ')
    print(func.imported_modules[0].get_source('uop'))

    a_np = np.random.randint(size=(fh, fw, fc), dtype='int8', low = -128, high = 127)

    a_nd = tvm.nd.array(a_np)
    gt_nd = tvm.nd.array(np.zeros((packed_shape[0] // 8, packed_shape[1] // 8, 8, 8), dtype='int8'))
    gt_func(a_nd, gt_nd)

    ctx = tvm.nd.TVMContext(13, 0)
    a_nd = tvm.nd.array(a_np, ctx)
    real_nd = tvm.nd.array(np.zeros((packed_shape[0] // 8, packed_shape[1] // 8, 8, 8), dtype='int8'), ctx)
    func(a_nd, real_nd)
    gt_np = gt_nd.asnumpy()
    real_np = real_nd.asnumpy()
    np.testing.assert_allclose(gt_np, real_np)
    print('test passed')