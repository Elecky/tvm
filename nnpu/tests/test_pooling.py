import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

# reduce max
def test():
    env = nnpu.get_env()
    nnpu.set_device(env)
    in_shape = (16,16,32)
    cell_shape = 4
    out_shape = (4,4,32)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    
    sph = ScheduleProcHelper()
    str_op = 'VAddMerge'
    a = tvm.placeholder(in_shape, dtype_w, 'a')
    a_buf, _1 = nnpu.utils.CopyHtoBuf(a, 'a', sph)

    k1 = tvm.reduce_axis((0, 4), 'k1')
    step1_buf = tvm.compute((16,4,32), 
                        lambda i,j,k: 
                         tvm.sum(a_buf[i,j*4+k1,k],axis=k1),
                       'step1_buf')
    sph.MarkScope(step1_buf)

    k2 = tvm.reduce_axis((0, 4), 'k2')
    step2_buf = tvm.compute((4,4,32), 
                        lambda i,j,k: 
                         tvm.sum(step1_buf[i*4+k2,j,k],axis=k2),
                       'step2_buf')
    sph.MarkScope(step2_buf)
    step2_host, step2_dram = nnpu.utils.CopyBufToH(step2_buf, 'step2',sph)

    s = tvm.create_schedule(step2_host.op)
    sph.Transform(s)

    ko, ki = s[step1_buf].split(step1_buf.op.reduce_axis[0], factor=1)
    xo,xi = s[step1_buf].split(step1_buf.op.axis[2], factor=16)
    yo,yi = s[step1_buf].split(step1_buf.op.axis[1], factor=1)
    zo,zi = s[step1_buf].split(step1_buf.op.axis[0], factor=1)
    #s[step1_buf].reorder( xo,ko,yo,zo,zi,yi,ki, xi)
    s[step1_buf].tensorize(ki, env.intrins.get(str_op,  mode='w'))

    ko1, ki1 = s[step2_buf].split(step2_buf.op.reduce_axis[0], factor=1)
    xo1,xi1 = s[step2_buf].split(step2_buf.op.axis[2], factor=16)
    yo1,yi1 = s[step2_buf].split(step2_buf.op.axis[1], factor=1)
    zo1,zi1 = s[step2_buf].split(step2_buf.op.axis[0], factor=1)
    #s[step2_buf].reorder( xo1,ko1,yo1,zo1,zi1,yi1,ki1, xi1)
    s[step2_buf].tensorize(ki1, env.intrins.get(str_op,  mode='w'))

    #tensorize

    print(nnpu.lower(s, [a, step2_host], simple_mode=True))
    exit()
    
    func = nnpu.build(s, [a, step2_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=in_shape, dtype=a.dtype, low = -4, high = 4)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros(out_shape, dtype=step2_host.dtype), ctx)

    func(a_nd, c_nd)
    print(str_op)
    print(c_nd.asnumpy())
    '''
    print("pooling-sum")
    '''

if __name__ == '__main__':
    test()