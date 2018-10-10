import nnpu
import tvm
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)

    shape = (8, 16)
    a_host = tvm.placeholder(shape, env.cfg['dtype_n'], 'a_host')
    a = tvm.compute(shape, lambda *i: a_host(*i), name='a')
    a_buf = tvm.compute(shape, lambda *i: a(*i), name='a_buf')
    
    vctr_shape = (1, 16)
    b_host = tvm.placeholder(vctr_shape, env.cfg['dtype_n'], 'b_host')
    b = tvm.compute(vctr_shape, lambda *i: b_host(*i), name='b')
    b_buf = tvm.compute(vctr_shape, lambda *i: b(*i), name='b_buf')

    dtype_w = env.cfg['dtype_w']
    out_shape = (8, 1)
    k = tvm.reduce_axis((0, 16), 'k')
    c_buf = tvm.compute(out_shape, 
                    lambda i, j: 
                        tvm.sum(a_buf[i, k].astype(dtype_w) * b_buf[j, k].astype(dtype_w), axis=k))
    c = tvm.compute(out_shape, lambda *i: c_buf(*i), name='c')
    c_host = tvm.compute(out_shape, lambda *i: c(*i), name='c_host')

    s = tvm.create_schedule(c_host.op)

    # mark variable scopes
    s[a].set_scope(env.dram_scope)
    s[b].set_scope(env.dram_scope)
    s[c].set_scope(env.dram_scope)

    s[a_buf].set_scope(env.uni_scratchpad_scope)
    s[b_buf].set_scope(env.uni_scratchpad_scope)
    s[c_buf].set_scope(env.uni_scratchpad_scope)

    #print(dir(s[b].op.body))

    # mark compiler pragmas
    s[a].pragma(s[a].op.axis[0], env.dma_copy_pragma)
    s[b].pragma(s[b].op.axis[0], env.dma_copy_pragma)
    s[c_host].pragma(s[c_host].op.axis[0], env.dma_copy_pragma)

    s[a_buf].pragma(s[a_buf].op.axis[0], env.scratchpad_ls)
    s[b_buf].pragma(s[b_buf].op.axis[0], env.scratchpad_ls)
    s[c].pragma(s[c].op.axis[0], env.scratchpad_ls)

    #s[a_buf].compute_at(s[b_buf], b_buf.op.axis[0])

    # tensorize
    #s[b_buf].tensorize(s[b_buf].op.axis[1], env.intrins.get('VEXP', mode='inc'))
    s[c_buf].tensorize(s[c_buf].op.axis[0], env.intrins.get('GEMM', shape=(8, 16, 1), mode='inc'))

    # build
    print(tvm.lower(s, [a_host, b_host, c_host], simple_mode=True))

    print(nnpu.lower(s, [a_host, b_host, c_host], simple_mode=True))
    #exit()
    func = nnpu.build(s, [a_host, b_host, c_host], 'nnpu', 'llvm', name='nnpu_exp')

    print('function built: ')
    #print(func.get_source())

    # prepare data
    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=shape, dtype=a_host.dtype, low = 0, high = 64)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=vctr_shape, dtype=b_host.dtype, low = 0, high = 64)
    b_nd = tvm.nd.array(b_np, ctx)

    c_nd = tvm.nd.array(np.zeros(out_shape).astype(c_host.dtype), ctx)

    # run
    func(a_nd, b_nd, c_nd)

    print('run finished')

    print('a=')
    print(a_np)
    print('b=')
    print(b_np)
    print('c=')
    print(c_nd.asnumpy())

    print('numpy ground truth is: ')
    gt = np.dot(a_np.astype(dtype_w), b_np.astype(dtype_w).transpose((1,0)))
    print(gt)

    np.testing.assert_allclose(c_nd.asnumpy(), gt)

if __name__ == '__main__':
    test()