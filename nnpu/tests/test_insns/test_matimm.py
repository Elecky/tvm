import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    a = tvm.placeholder((16,16), env.cfg['dtype_n'], 'a')
    sph = ScheduleProcHelper()
    Imm = tvm.const(7, env.cfg['dtype_n'])
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    add_buf = tvm.compute((16,16), lambda i,j: Imm+a_buf[i][j], 'add_buf')
    sph.MarkScope(add_buf)
    add_host, add_dram = nnpu.utils.CopyBufToH(add_buf, 'add', sph)

    dtype_w = env.cfg['dtype_w']
    mul_buf = tvm.compute((16,16), lambda i,j: a_buf[i][j].astype(dtype_w) * Imm.astype(dtype_w), 
                          'mul_buf')
    sph.MarkScope(mul_buf)
    mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    rsub_buf = tvm.compute((16,16), lambda i,j: Imm-a_buf[i][j], 'rsub_buf')
    sph.MarkScope(rsub_buf)
    rsub_host, rsub_dram = nnpu.utils.CopyBufToH(rsub_buf, 'rsub', sph)

    s = tvm.create_schedule([add_host.op,mul_host.op,rsub_host.op])
    sph.Transform(s)
    s[add_buf].tensorize(s[add_buf].op.axis[0], env.intrins.get('MAddI', 
                            shape=(16,16), imm_value=Imm.value, mode='n'))
    s[mul_buf].tensorize(s[mul_buf].op.axis[0], env.intrins.get('MMulI', 
                            shape=(16,16), imm_value=Imm.value, mode='inc'))
    s[rsub_buf].tensorize(s[rsub_buf].op.axis[0], env.intrins.get('ISubM', 
                            shape=(16,16), imm_value=Imm.value, mode='n'))
    print(nnpu.lower(s, [a,add_host,mul_host,rsub_host], simple_mode=True))
    func = nnpu.build(s, [a,add_host,mul_host,rsub_host], 'nnpu', 'llvm', name='nnpu_vmuli')
    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(16,16), dtype=a.dtype, low = 3, high = 100)
    a_nd = tvm.nd.array(a_np, ctx)

    add_nd = tvm.nd.array(np.zeros((16,16)).astype(add_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros((16,16)).astype(mul_host.dtype), ctx)
    rsub_nd = tvm.nd.array(np.zeros((16,16)).astype(rsub_host.dtype), ctx)
    func(a_nd, add_nd,mul_nd,rsub_nd)

    print(a_nd.asnumpy())
    print('add result is: ')
    print(add_nd.asnumpy())
    np.testing.assert_allclose(add_nd.asnumpy(), a_np + Imm.value)
    print('mul result is: ')
    print(mul_nd.asnumpy())
    np.testing.assert_allclose(mul_nd.asnumpy(), a_np.astype(dtype_w) * Imm.value)
    print('rsub result is: ')
    print(rsub_nd.asnumpy())
    np.testing.assert_allclose(rsub_nd.asnumpy(), Imm.value - a_np )
    print('test passed')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test of NNPU Op')
    parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
    args = parser.parse_args()

    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    test()