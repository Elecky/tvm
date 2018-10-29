import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

with ScheduleProcHelper():
    env = nnpu.get_env()
    nnpu.set_device(env)

    t = 10  # time 
    n = 16  # input depth
    m = 16  # output depth
    x_shape = (t, n)
    h_shape = (t, m)
    w_shape = (m, n)
    u_shape = (m, m)
    b_shape = (m, )
    gemm_shape = (16, 16, 1)

    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    x = tvm.placeholder(x_shape, dtype_n, 'x')
    x_buf, _ = nnpu.utils.CopyHtoBuf(x, 'x')

    w = tvm.placeholder(w_shape, dtype_n, 'w')
    w_buf, _ = nnpu.utils.CopyHtoBuf(w, 'w')

    u = tvm.placeholder(u_shape, dtype_w, 'u')
    u_buf, _ = nnpu.utils.CopyHtoBuf(u, 'u')

    b = tvm.placeholder(b_shape, dtype_w, 'b')
    b_buf, _ = nnpu.utils.CopyHtoBuf(b, 'b')

    h_state = tvm.placeholder(h_shape, dtype_w, 'h')
    nnpu.utils.MarkScope(h_state)

    h_init = tvm.placeholder((1, h_shape[1]), dtype_w, 'h_init')
    h_init_buf, _ = nnpu.utils.CopyHtoBuf(h_init, 'h_init')

    #h_init = tvm.compute((1, h_shape[1]), lambda _, i: h_init_buf[i])
    #nnpu.utils.MarkScope(h_init)

    k = tvm.reduce_axis((0, n), 'k0')
    s_update_1 = tvm.compute(h_shape, 
                        lambda t, i: tvm.sum(w_buf[i, k].astype(dtype_w) * x_buf[t, k].astype(dtype_w),
                                            axis=k),
                        's1')
    nnpu.utils.MarkScope(s_update_1)
    k = tvm.reduce_axis((0, m), 'k1')
    s_update_2 = tvm.compute(h_shape, 
                        lambda t, i: tvm.sum(u_buf[i, k] * h_state[t - 1, k], axis=k),
                        's2')
    nnpu.utils.MarkScope(s_update_2)
    s_update_3 = tvm.compute(h_shape,
                        lambda t, i: s_update_1[t, i] + s_update_2[t, i], 
                        's3')
    nnpu.utils.MarkScope(s_update_3)
    s_update_4 = tvm.compute(h_shape,
                        lambda t, i: s_update_3[t, i] + b_buf[i],
                        's4')
    nnpu.utils.MarkScope(s_update_4)
    s_scan = tvm.scan(h_init_buf, s_update_4, h_state, inputs=[x_buf])
    nnpu.utils.MarkScope(s_scan)

    #res = nnpu.utils.reshape(s_scan, h_shape)
    #res_host, _ = nnpu.utils.CopyBufToH(res, 'sc')
    s = nnpu.create_schedule(s_scan.op)
    # tensorize
    s[s_update_1].tensorize(s_update_1.op.axis[1], 
                            env.intrins.get('GEMM', shape=gemm_shape, mode='inc', reduce=True))
    #s[s_update_2].tensorize(s_update_2.op.axis[1],
    #                        env.intrins.get('GEMM', shape=gemm_shape, mode='w', reduce=True))
    s[s_update_3].tensorize(s_update_3.op.axis[1],
                            env.intrins.get('VAddV', mode='w'))
    #s[s_update_4].tensorize(s_update_4.op.axis[1],
    #                        env.intrins.get('VAddV', mode='w'))
    print(tvm.lower(s, [x, w, u, b, h_init, s_scan], simple_mode=True))