import nnpu
import tvm
import topi

def test():
    env = nnpu.get_env()

    a = tvm.placeholder((16, 16), 'int16', 'a')
    b = tvm.placeholder((16, 1), 'int16', 'b')
    c = topi.squeeze(b, axis=1)
    k = tvm.reduce_axis((0,16), 'k')
    out = tvm.compute((16, ), lambda i: tvm.sum(a[i, k] * c[k], axis = k), 'out')

    s = tvm.create_schedule(out.op)

    print(tvm.lower(s, [a, b, out], simple_mode=True))

if __name__ == '__main__':
    test()