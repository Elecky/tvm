import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

def test():
    env = nnpu.get_env()
    nnpu.set_device(env)
    a = tvm.placeholder((16, ), env.cfg['dtype_w'], 'a')
    sph = ScheduleProcHelper()
    Imm = tvm.const(5, env.cfg['dtype_w'])
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    #c_buf = tvm.compute((16, ), lambda i: tvm.select(a_buf[i]>Imm,a_buf[i],Imm), 'c_buf')
    c_buf = tvm.compute((16, ), lambda i: Imm+a_buf[i], 'c_buf')
    sph.MarkScope(c_buf)
    c_host, c_dram = nnpu.utils.CopyBufToH(c_buf, 'c', sph)

    sub_buf = tvm.compute((16, ), lambda i: a_buf[i] - Imm, 'sub_buf')
    sph.MarkScope(sub_buf)
    sub_host, sub_dram = nnpu.utils.CopyBufToH(sub_buf, 'sub', sph)

    mul_buf = tvm.compute((16, ), lambda i: a_buf[i] * Imm, 'mul_buf')
    sph.MarkScope(mul_buf)
    mul_host, mul_dram = nnpu.utils.CopyBufToH(mul_buf, 'mul', sph)

    div_buf = tvm.compute((16, ), lambda i: a_buf[i] / Imm, 'div_buf')
    sph.MarkScope(div_buf)
    div_host, div_dram = nnpu.utils.CopyBufToH(div_buf, 'div', sph)

    gtm_buf = tvm.compute((16, ), lambda i: tvm.select(a_buf[i] > Imm, a_buf[i], Imm), 'gtm_buf')
    sph.MarkScope(gtm_buf)
    gtm_host, gtm_dram = nnpu.utils.CopyBufToH(gtm_buf, 'gtm', sph)

    s = tvm.create_schedule([c_host.op, sub_host.op, mul_host.op, div_host.op, gtm_host.op])
    sph.Transform(s)
    s[c_buf].tensorize(s[c_buf].op.axis[0], env.intrins.get('VAddI', imm_value=Imm.value,mode='w'))
    s[sub_buf].tensorize(s[sub_buf].op.axis[0], env.intrins.get('VSubI', imm_value=Imm.value,mode='w'))
    s[mul_buf].tensorize(s[mul_buf].op.axis[0], env.intrins.get('VMulI', imm_value=Imm.value,mode='w'))
    s[div_buf].tensorize(s[div_buf].op.axis[0], env.intrins.get('VDivI', imm_value=Imm.value,mode='w'))
    s[gtm_buf].tensorize(s[gtm_buf].op.axis[0], env.intrins.get('VGTMI', imm_value=Imm.value,mode='w'))

    print(nnpu.lower(s, [a,c_host,sub_host,mul_host,div_host,gtm_host], simple_mode=True))
    func = nnpu.build(s, [a,c_host,sub_host,mul_host,div_host,gtm_host], 'nnpu', 'llvm', name='nnpu_vmuli')


    ctx = tvm.nd.TVMContext(13, 0)

    a_np = np.random.randint(size=(16, ), dtype=a.dtype, low = 3, high = 23)
    #a_np = np.random.random(size=shape).astype(a_host.dtype)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    sub_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    mul_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    div_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)
    gtm_nd = tvm.nd.array(np.zeros((16, )).astype(c_host.dtype), ctx)

    func(a_nd, c_nd, sub_nd, mul_nd, div_nd, gtm_nd)
    print(a_nd.asnumpy())
    print('add result is: ')
    print(c_nd.asnumpy())
    print('numpy ground truth is: ')
    gt = a_np + Imm.value
    print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)

    print('sub result is: ')
    print(sub_nd.asnumpy())
    np.testing.assert_allclose(sub_nd.asnumpy(), a_np - Imm.value)

    print('mul result is: ')
    print(mul_nd.asnumpy())
    np.testing.assert_allclose(mul_nd.asnumpy(), a_np * Imm.value)

    print('div result is: ')
    print(div_nd.asnumpy())
    np.testing.assert_allclose(div_nd.asnumpy(), a_np / Imm.value)

    print('gtm result is: ')
    print(gtm_nd.asnumpy())
    #np.testing.assert_allclose(gtm_nd.asnumpy(), a_np  Imm.value)

if __name__ == '__main__':
    test()