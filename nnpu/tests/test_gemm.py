import nnpu
import tvm
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env, type='S1')

    shape = (16, 16)
    a_host = tvm.placeholder(shape, env.cfg['dtype_n'], 'a_host')
    a = tvm.compute(shape, lambda *i: a_host(*i), name='a')
    a_buf = tvm.compute(shape, lambda *i: a(*i), name='a_buf')
    
    vctr_shape = (16, )
    b_host = tvm.placeholder(vctr_shape, env.cfg['dtype_n'], 'b_host')
    b = tvm.compute(vctr_shape, lambda *i: b_host(*i), name='b')
    b_buf = tvm.compute(vctr_shape, lambda *i: b(*i), name='b_buf')

    dtype_w = env.cfg['dtype_w']
    
    out_shape = (16,)
    k = tvm.reduce_axis((0, 16), 'k')
    c_buf = tvm.compute(out_shape, 
                    lambda i: 
                        tvm.sum(a_buf[i, k].astype(dtype_w) * b_buf[k].astype(dtype_w), axis=k))
    
    bias_host = tvm.placeholder(out_shape, env.cfg['dtype_w'], 'bias_host')
    bias = tvm.compute(out_shape, lambda *i: bias_host(*i), 'bias')
    bias_buf = tvm.compute(out_shape, lambda *i: bias(*i), 'bias_buf')
    #c = tvm.compute(out_shape, lambda *i: c_buf(*i), name='c')
    #c_host = tvm.compute(out_shape, lambda *i: c(*i), name='c_host')

    out_buf = tvm.compute(out_shape, lambda i: c_buf[i] + bias_buf[i], 'out_buf')
    out = tvm.compute(out_shape, lambda *i: out_buf(*i), 'out')
    out_host = tvm.compute(out_shape, lambda *i: out(*i), 'out_host')

    s = tvm.create_schedule(out_host.op)

    # mark variable scopes
    s[a].set_scope(env.dram_scope)
    s[b].set_scope(env.dram_scope)
    s[bias].set_scope(env.dram_scope)
    s[out].set_scope(env.dram_scope)

    s[a_buf].set_scope(env.uni_scratchpad_scope)
    s[b_buf].set_scope(env.uni_scratchpad_scope)
    s[c_buf].set_scope(env.uni_scratchpad_scope)
    s[bias_buf].set_scope(env.uni_scratchpad_scope)
    s[out_buf].set_scope(env.uni_scratchpad_scope)

    #print(dir(s[b].op.body))

    # mark compiler pragmas
    s[a].pragma(s[a].op.axis[0], env.dma_copy_pragma)
    s[b].pragma(s[b].op.axis[0], env.dma_copy_pragma)
    s[bias].pragma(s[bias].op.axis[0], env.dma_copy_pragma)
    s[out_host].pragma(s[out_host].op.axis[0], env.dma_copy_pragma)

    s[a_buf].pragma(s[a_buf].op.axis[0], env.scratchpad_ls)
    s[b_buf].pragma(s[b_buf].op.axis[0], env.scratchpad_ls)
    s[bias_buf].pragma(s[bias_buf].op.axis[0], env.scratchpad_ls)
    s[out].pragma(s[out].op.axis[0], env.scratchpad_ls)

    #s[a_buf].compute_at(s[b_buf], b_buf.op.axis[0])

    # tensorize
    #s[b_buf].tensorize(s[b_buf].op.axis[1], env.intrins.get('VEXP', mode='inc'))
    s[c_buf].tensorize(s[c_buf].op.axis[0], env.intrins.get('GEMM', shape=(16, 16, 1), 
                        mode='inc', reduce=True))
    #outer, inner = out_buf.op.axis
    #s[out_buf].reorder(inner, outer)
    #print(outer)
    #print(tvm.lower(s, [a_host, b_host, bias_host, out_host], simple_mode=True))
    s[out_buf].tensorize(s[out_buf].op.axis[0], env.intrins.get('VAddV', mode='w'))

    # build
    print(tvm.lower(s, [a_host, b_host, bias_host, out_host], simple_mode=True))

    print(nnpu.lower(s, [a_host, b_host, bias_host, out_host], simple_mode=True))
    #exit()
    func = nnpu.build(s, [a_host, b_host, bias_host, out_host], 'nnpu', 'llvm', name='nnpu_exp')

    print('function built: ')
    #print(func.get_source())

    # prepare data
    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=shape, dtype=a_host.dtype, low = 0, high = 64)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    b_np = np.random.randint(size=vctr_shape, dtype=b_host.dtype, low = 0, high = 64)
    #b_np = np.random.random(size=vctr_shape).astype(b_host.dtype)
    b_nd = tvm.nd.array(b_np, ctx)

    bias_np = np.random.randint(size=out_shape, dtype=bias_host.dtype, low = 0, high = 10000)
    #bias_np = np.random.random(size=out_shape).astype(bias_host.dtype)
    bias_nd = tvm.nd.array(bias_np, ctx)

    out_nd = tvm.nd.array(np.zeros(out_shape).astype(out_host.dtype), ctx)

    # run
    func(a_nd, b_nd, bias_nd, out_nd)

    print('run finished')

    print('a=')
    print(a_np)
    print('b=')
    print(b_np)
    print('bias=')
    print(bias_np)
    print('out=')
    print(out_nd.asnumpy())

    print('numpy ground truth is: ')
    gt = np.dot(a_np.astype(dtype_w), b_np.astype(dtype_w)) + bias_np
    #gt = np.greater(np.dot(a_np.astype(dtype_w), b_np.astype(dtype_w)), bias_np)
    print(gt)

    np.testing.assert_allclose(out_nd.asnumpy(), gt)

if __name__ == '__main__':
    test()