import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

with ScheduleProcHelper():
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (16, 32)  # (16, 32) tiled to (16, 2, 16)
    a = tvm.placeholder(shape, env.cfg['dtype_n'], 'a')
    b = tvm.placeholder(shape, env.cfg['dtype_n'], 'b')

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a')
    b_buf, b_dram = nnpu.utils.CopyHtoBuf(b, 'b')
    
    sum_buf = tvm.compute(shape, lambda *i: a_buf(*i) + b_buf(*i), 'sum_buf')
    nnpu.utils.MarkScope(sum_buf)
    sum_host, sum_dram = nnpu.utils.CopyBufToH(sum_buf, 'sum')

    s = nnpu.create_schedule(sum_host.op)
    # tensorize
    x = sum_buf.op.axis[0]
    y,z = s[sum_buf].split(sum_buf.op.axis[1], factor=16)
    s[sum_buf].reorder(y, x, z)
    s[sum_buf].tensorize(x, env.intrins.get('MAddM', shape=(16, 16), mode='n'))

    print(nnpu.lower(s, [a, b, sum_host], simple_mode=True))

    func = nnpu.build(s, [a, b, sum_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape, dtype=a.dtype, low = -32, high = 32)
    a_nd = tvm.nd.array(a_np, ctx)
    b_np = np.random.randint(size=shape, dtype=b.dtype, low = -32, high = 32)
    b_nd = tvm.nd.array(b_np, ctx)

    sum_nd = tvm.nd.array(np.zeros(shape, dtype=sum_host.dtype), ctx)

    func(a_nd, b_nd, sum_nd)

    gt = a_np + b_np
    np.testing.assert_allclose(sum_nd.asnumpy(), gt)