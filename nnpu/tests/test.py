import nnpu
import tvm

def test():
    env = nnpu.get_env()

    a = tvm.placeholder((16,), 'float16', 'a')
    a_buf = tvm.compute((16,), lambda *i: a(*i), name='a_buf')
    b_buf = tvm.compute((16,), lambda i: tvm.exp(a_buf[i]), name='b_buf')
    one = tvm.const(1, dtype='float16')
    c_buf = tvm.compute((16,), lambda i: b_buf[i] + one, name='c_buf')
    d_buf = tvm.compute((16,), lambda i: b_buf[i] / c_buf[i], name='d_buf')

    d = tvm.compute((16,), lambda *i: d_buf(*i), name='d')

    s = tvm.create_schedule(d.op)
    print(tvm.lower(s, [a, d], simple_mode=True))

    s[a_buf].set_scope(env.uni_scratchpad_scope)
    s[b_buf].set_scope(env.uni_scratchpad_scope)
    s[c_buf].set_scope(env.uni_scratchpad_scope)
    s[d_buf].set_scope(env.uni_scratchpad_scope)

    print(tvm.lower(s, [a, d], simple_mode=True))

    s[b_buf].tensorize(s[b_buf].op.axis[0], env.intrins['VEXP'])
    s[c_buf].tensorize(s[c_buf].op.axis[0], env.intrins['VAS'])
    s[d_buf].tensorize(s[d_buf].op.axis[0], env.intrins['VDV'])

    print(tvm.lower(s, [a, d], simple_mode=True))

if __name__ == '__main__':
    test()