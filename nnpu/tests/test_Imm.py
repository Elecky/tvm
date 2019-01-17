import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env, type='S1')
    a = tvm.placeholder((16, ), env.cfg['dtype_w'], 'a')
    sph = ScheduleProcHelper()
    Imm = tvm.const(5, env.cfg['dtype_w'])
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    #c_buf = tvm.compute((16, ), lambda i: tvm.select(a_buf[i]>Imm,a_buf[i],Imm), 'c_buf')
    c_buf = tvm.compute((16, ), lambda i: Imm+a_buf[i], 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    sub_buf = tvm.compute((16, ), lambda i: a_buf[i] - Imm , 'sub_buf')
    sph.MarkScope(sub_buf)
    sub_host, sub_dram = nnpu.utils.CopyBufToH(sub_buf, 'sub', sph)

    mul_buf = tvm.compute((16, ), lambda i: a_buf[i] * Imm, 'mul_buf')
    sph.MarkScope(mul_buf)
    mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    div_buf = tvm.compute((16, ), lambda i: a_buf[i] / Imm, 'rdiv_buf')
    sph.MarkScope(div_buf)
    div_host, div_dram = nnpu.utils.CopyBufToH(div_buf, 'rdiv', sph)

    gtm_buf = tvm.compute((16, ), lambda i: tvm.select(a_buf[i] > Imm, a_buf[i], Imm), 'gtm_buf')
    sph.MarkScope(gtm_buf)
    gtm_host, gtm_dram = nnpu.utils.CopyBufToH(gtm_buf, 'gtm', sph)

    rsub_buf = tvm.compute((16, ), lambda i: Imm-a_buf[i], 'rsub_buf')
    sph.MarkScope(rsub_buf)
    rsub_host, rsub_dram = nnpu.utils.CopyBufToH(rsub_buf, 'rsub', sph)


    s = tvm.create_schedule([c_host.op, sub_host.op, mul_host.op, div_host.op, gtm_host.op,rsub_host.op])
    sph.Transform(s)
    s[c_buf].tensorize(s[c_buf].op.axis[0], env.intrins.get('VAddI', imm_value=Imm.value,mode='w'))
    s[sub_buf].tensorize(s[sub_buf].op.axis[0], env.intrins.get('VSubI', imm_value=Imm.value,mode='w'))
    s[mul_buf].tensorize(s[mul_buf].op.axis[0], env.intrins.get('VMulI', imm_value=Imm.value,mode='w'))
    s[div_buf].tensorize(s[div_buf].op.axis[0], env.intrins.get('VDivI', imm_value=Imm.value,mode='w'))
    s[gtm_buf].tensorize(s[gtm_buf].op.axis[0], env.intrins.get('VGTMI', imm_value=Imm.value,mode='w'))
    s[rsub_buf].tensorize(s[rsub_buf].op.axis[0], env.intrins.get('ISubV', imm_value=Imm.value,mode='w'))
    print(nnpu.lower(s, [a,c_host,sub_host,mul_host,div_host,gtm_host,rsub_host], simple_mode=True))
    func = nnpu.build(s, [a,c_host,sub_host,mul_host,div_host,gtm_host,rsub_host], 'nnpu', 'llvm', name='nnpu_vmuli')


    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(16, ), dtype=a.dtype, low = 3, high = 122)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    sub_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    div_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    gtm_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    rsub_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    func(a_nd, c_nd, sub_nd, mul_nd, div_nd, gtm_nd,rsub_nd)
    print('a = ')
    print(a_nd.asnumpy())
    print('a + {0} = '.format(Imm.value))
    print(c_nd.asnumpy())
    print('numpy ground truth =')
    gt = a_np + Imm.value
    print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)

    print('a - {0} = '.format(Imm.value))
    print(sub_nd.asnumpy())
    np.testing.assert_allclose(sub_nd.asnumpy(), a_np - Imm.value)

    print('a * {0} = '.format(Imm.value))
    print(mul_nd.asnumpy())
    np.testing.assert_allclose(mul_nd.asnumpy(), a_np * Imm.value)

    print('a / {0} = '.format(Imm.value))
    print(div_nd.asnumpy())
    np.testing.assert_allclose(div_nd.asnumpy(), a_np / Imm.value)

    print('a > {0} ? a : {0} = '.format(Imm.value))
    print(gtm_nd.asnumpy())
    #np.testing.assert_allclose(gtm_nd.asnumpy(), a_np  Imm.value)
    print('{0} - a = '.format(Imm.value))
    print(rsub_nd.asnumpy())
    np.testing.assert_allclose(rsub_nd.asnumpy(), Imm.value-a_np)
if __name__ == '__main__':
    test()