import nnpu
import tvm
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (2, 16)
    a_host = tvm.placeholder(shape, env.cfg['dtype_n'], 'a_host')
    print('a host '+str(a_host))
    a = tvm.compute(shape, lambda *i: a_host(*i), name='a')
    a_buf = tvm.compute(shape, lambda *i: a(*i), name='a_buf')
    b_buf = tvm.compute(shape, lambda i, j: tvm.log(a_buf[i, j].astype(env.cfg['dtype_w'])), name='b_buf')
    b = tvm.compute(shape, lambda *i: b_buf(*i), name='b')
    b_host = tvm.compute(shape, lambda *i: b(*i), name='b_host')

    s = tvm.create_schedule(b_host.op)

    # mark variable scopes
    s[a].set_scope(env.dram_scope)
    s[b].set_scope(env.dram_scope)

    s[a_buf].set_scope(env.uni_scratchpad_scope)
    s[b_buf].set_scope(env.uni_scratchpad_scope)

    #print
    # (dir(s[b].op.body))

    # mark compiler pragmas
    s[a].pragma(s[a].op.axis[0], env.dma_copy_pragma)
    s[b_host].pragma(s[b_host].op.axis[0], env.dma_copy_pragma)

    s[a_buf].pragma(s[a_buf].op.axis[0], env.scratchpad_ls)
    s[b].pragma(s[b].op.axis[0], env.scratchpad_ls)

    s[a_buf].compute_at(s[b_buf], b_buf.op.axis[0])

    # tensorize
    s[b_buf].tensorize(s[b_buf].op.axis[1], env.intrins.get('VLOG', mode='inc'))

    # build
    print(tvm.lower(s, [a_host, b_host], simple_mode=True))
    print(nnpu.lower(s, [a_host, b_host], simple_mode=True))
    #exit()
    func = nnpu.build(s, [a_host, b_host], 'nnpu', 'llvm', name='nnpu_log')

    print('function built: ')
    #print(func.get_source())

    # prepare data
    ctx = tvm.nd.TVMContext(13, 0)#???
    print ('i want to know:')
    print (ctx.exist)
    a_np = np.random.randint(size=shape, dtype=a_host.dtype, low = 1, high = 20)
    a_nd = tvm.nd.array(a_np, ctx)
    b_nd = tvm.nd.array(np.zeros(shape).astype(b_host.dtype), ctx)

    # run
    func(a_nd, b_nd)

    print('run finished')

    b_np = b_nd.asnumpy()
    print('a=')
    print(a_np)
    print('b=')
    print(b_np)
    print('ground truth =')
    gt = np.log(a_np, dtype=b_host.dtype)
    print(gt)
    np.testing.assert_allclose(b_np, gt)

if __name__ == '__main__':
    test()