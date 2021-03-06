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

with ScheduleProcHelper():
    env = nnpu.get_env()
    shape = (32, 48)  # (32, 32) tiled to (2, 16, 2, 16)
    insn_shape = (16, 16)
    assert shape[0] % insn_shape[0] == 0, 'error'
    assert shape[1] % insn_shape[1] == 0, 'error'

    a = tvm.placeholder(shape, env.cfg['dtype_n'], 'a')
    b = tvm.placeholder(shape, env.cfg['dtype_n'], 'b')
    dtype_w = env.cfg['dtype_w']

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a')
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b')
    
    sum_buf = tvm.compute(shape, lambda *i: a_buf(*i) + b_buf(*i), 'sum_buf')
    nnpu.utils.MarkScope(sum_buf)
    sum_host, sum_dram = nnpu.utils.CopyBufToH(sum_buf, 'sum')

    mul_buf = tvm.compute(shape, lambda *i: a_buf(*i).astype(dtype_w) * b_buf(*i).astype(dtype_w), 'mul_buf')
    nnpu.utils.MarkScope(mul_buf)
    mul_host, _ = nnpu.utils.CopyBufToH(mul_buf, 'mul')

    s = nnpu.create_schedule([sum_host.op, mul_host.op])
    # tensorize
    xo, xi = s[sum_buf].split(sum_buf.op.axis[0], factor=insn_shape[0])
    yo, yi = s[sum_buf].split(sum_buf.op.axis[1], factor=insn_shape[1])
    s[sum_buf].reorder(xo, yo, xi, yi)
    s[sum_buf].tensorize(xi, env.intrins.get('MAddM', shape=insn_shape, mode='n'))

    # xo, xi = s[sum_buf].mul_buf(mul_buf.op.axis[0], factor=insn_shape[0])
    # yo, yi = s[sum_buf].split(sum_buf.op.axis[1], factor=insn_shape[1])
    # s[sum_buf].reorder(xo, yo, xi, yi)
    s[mul_buf].tile(mul_buf.op.axis[0], mul_buf.op.axis[1], insn_shape[0], insn_shape[1])
    s[mul_buf].tensorize(s[mul_buf].leaf_iter_vars[2], env.intrins.get('MMulM', shape=insn_shape, mode='inc'))

    print(nnpu.lower(s, [a, b, sum_host, mul_host], simple_mode=True))

    func = nnpu.build(s, [a, b, sum_host, mul_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape, dtype=a.dtype, low = -32, high = 32)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=shape, dtype=b.dtype, low = -32, high = 32)
    b_nd = tvm.nd.array(b_np, ctx)

    sum_nd = tvm.nd.array(np.zeros(shape, dtype=sum_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros(shape, dtype=mul_host.dtype), ctx)

    func(a_nd, b_nd, sum_nd, mul_nd)

    gt = a_np + b_np
    np.testing.assert_allclose(sum_nd.asnumpy(), gt)
    gt = a_np.astype(dtype_w) * b_np.astype(dtype_w)
    np.testing.assert_allclose(mul_nd.asnumpy(), gt)
    print('test finished')