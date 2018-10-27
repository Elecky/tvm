import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np


def mean_pooling(inshape,outshape,cell_shape,innp,outdetype):
  ret=np.zeros(outshape, dtype=outdetype)
  for w in range(outshape[0]):
    for h in range(outshape[1]):
      for j in range(cell_shape):
        for k in range(cell_shape):
          ret[w][h]=ret[w][h]+innp[w*4+j][h*4+k]
  di=cell_shape*cell_shape
  for w in range(outshape[0]):
    for h in range(outshape[1]):
      ret[w][h]=ret[w][h]/di
  return ret    


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
    step1_buf = tvm.compute((4,16,32), 
                        lambda j,i,k: 
                         tvm.sum(a_buf[i,j*4+k1,k],axis=k1),
                       'step1_buf')
    sph.MarkScope(step1_buf)

    k1 = tvm.reduce_axis((0, 4), 'k2')
    step2_buf = tvm.compute((4,4,32), 
                        lambda i,j,k: 
                         tvm.sum(step1_buf[j,i*4+k1,k],axis=k1),
                       'step2_buf')
    sph.MarkScope(step2_buf)
    

    Imm = tvm.const(cell_shape*cell_shape, env.cfg['dtype_w'])
    step3_buf = tvm.compute((4,4,32), 
                        lambda i,j,k: 
                         step2_buf[i,j,k]/Imm,
                       'step3_buf')
    sph.MarkScope(step3_buf)

    step3_host, step3_dram = nnpu.utils.CopyBufToH(step3_buf, 'step3',sph)
    s = tvm.create_schedule(step3_host.op)
    sph.Transform(s)


    #tensorize
    ko, ki = s[step1_buf].split(step1_buf.op.reduce_axis[0], factor=1)
    xo,xi = s[step1_buf].split(step1_buf.op.axis[2], factor=16)
    s[step1_buf].reorder( step1_buf.op.axis[1],step1_buf.op.axis[0],xo,ko,ki,xi)
    s[step1_buf].tensorize(ki, env.intrins.get(str_op,  mode='w'))

    ko1, ki1 = s[step2_buf].split(step2_buf.op.reduce_axis[0], factor=1)
    xo1,xi1 = s[step2_buf].split(step2_buf.op.axis[2], factor=16)
    s[step2_buf].reorder( step2_buf.op.axis[0],step2_buf.op.axis[1],xo1,ko1,ki1, xi1)
    s[step2_buf].tensorize(ki1, env.intrins.get(str_op,  mode='w'))

    xo2,xi2 = s[step3_buf].split(step3_buf.op.axis[2], factor=16)
    s[step3_buf].reorder( step3_buf.op.axis[0],step3_buf.op.axis[1],xo2,xi2)
    s[step3_buf].tensorize(xi2, env.intrins.get('VDivI',  mode='w'))

    
    

    print(nnpu.lower(s, [a, step3_host], simple_mode=True))
    exit()
    func = nnpu.build(s, [a, step3_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=in_shape, dtype=a.dtype, low = -8, high = 8)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros(out_shape, dtype=step2_host.dtype), ctx)

    func(a_nd, c_nd)
    print("pooling-sum")
    print(c_nd.asnumpy())
    
    print("nppooling-sum")
    print(mean_pooling(in_shape,out_shape,cell_shape,a_np,a.dtype))
    

if __name__ == '__main__':
    test()