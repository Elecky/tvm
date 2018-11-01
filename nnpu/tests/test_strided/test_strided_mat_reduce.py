import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

with ScheduleProcHelper():
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (32, 128)  # (32, 64) -> (32, )
    rshape = (16, 16)  # the shape that MReduceSum insn accepts
    assert shape[0] % rshape[0] == 0, 'height must be divisible to {0}'.format(rshape[0])
    assert shape[0] % env.cfg['vector_unit']['size'] == 0, \
        'height must be divisible to {0}'.format(env.cfg['vector_unit']['size'])
    assert shape[1] % rshape[1] == 0, 'width must be divisible to {0}'.format(rshape[0])
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w'],
    a = tvm.placeholder(shape, dtype_n, 'a')

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a')
    
    k = tvm.reduce_axis((0, shape[1]), 'k0')
    re_shape = (shape[0], )
    re_buf = tvm.compute(re_shape, 
                         lambda i: tvm.sum(a_buf[i, k].astype(dtype_w), axis=k), 're_buf')
    nnpu.utils.MarkScope(re_buf, 'acc')
    
    res_buf = nnpu.utils.CopyAccToBuf(re_buf, 'res')

    res_host, _ = nnpu.utils.CopyBufToH(res_buf, 'res')

    s = nnpu.create_schedule(res_host.op)
    # tensorize
    xo, xi = s[re_buf].split(re_buf.op.axis[0], rshape[0])
    ro, ri = s[re_buf].split(re_buf.op.reduce_axis[0], rshape[1])
    s[re_buf].reorder(xo, ro, xi, ri)
    s[re_buf].tensorize(xi, env.intrins.get('MReduceSumRow', mode='inc', 
                                            shape=rshape, scope_out='acc'))

    print(nnpu.lower(s, [a, res_host], simple_mode=True))
    #exit(0)
    func = nnpu.build(s, [a, res_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=shape, dtype=a.dtype, low = -128, high = 127)
    a_nd = tvm.nd.array(a_np, ctx)

    res_nd = tvm.nd.array(np.zeros((shape[0], ), dtype=res_host.dtype), ctx)

    func(a_nd, res_nd)
    print('a = ')
    #print(a_np)
    print('reduced = ')
    print(res_nd.asnumpy())
    gt = np.sum(a_np, axis=1, dtype=dtype_w)
    print('ground truth=')
    print(gt)
    np.testing.assert_allclose(res_nd.asnumpy(), gt)