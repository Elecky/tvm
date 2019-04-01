import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np


def max_pooling(inshape,outshape,cell_shape,innp,outdetype):
  ret=np.zeros(outshape, dtype=outdetype)
  for w in range(outshape[0]):
    for h in range(outshape[1]):
      for j in range(cell_shape):
        for k in range(cell_shape):
          for l in range(outshape[2]):
            ret[w][h][l]=max(ret[w][h][l],innp[w*cell_shape+j][h*cell_shape+k][l])
  return ret    


# reduce max
def test():
    env = nnpu.get_env()
    nnpu.set_dump(False)
    in_shape = (20,20,32)
    cell_shape = 5
    out_shape = (in_shape[0] // cell_shape,in_shape[1] // cell_shape,in_shape[2])
    reduce_shap=(0,cell_shape)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    
    sph = ScheduleProcHelper()
    str_op = 'VGTMMerge'
    a = tvm.placeholder(in_shape, dtype_w, 'a')
    a_buf, _1 = nnpu.utils.CopyHtoBuf(a, 'a', sph)

    k1 = tvm.reduce_axis(reduce_shap, 'k1')
    step1_buf = tvm.compute((in_shape[0],in_shape[1]//cell_shape,in_shape[2]), 
                        lambda i,j,k: 
                         tvm.max(a_buf[i,j*cell_shape+k1,k],axis=k1),
                       'step1_buf')
    sph.MarkScope(step1_buf)

    k1 = tvm.reduce_axis(reduce_shap, 'k2')
    step2_buf = tvm.compute(out_shape, 
                        lambda i,j,k: 
                         tvm.max(step1_buf[i*cell_shape+k1,j,k],axis=k1),
                       'step2_buf')
    sph.MarkScope(step2_buf)
    
    step2_host, step2_dram = nnpu.utils.CopyBufToH(step2_buf, 'step2',sph)
    s = tvm.create_schedule(step2_host.op)
    sph.Transform(s)


    #tensorize
    ko, ki = s[step1_buf].split(step1_buf.op.reduce_axis[0], factor=1)
    xo,xi = s[step1_buf].split(step1_buf.op.axis[2], factor=16)
    s[step1_buf].reorder( step1_buf.op.axis[0],step1_buf.op.axis[1],xo,ko,ki,xi)
    s[step1_buf].tensorize(ki, env.intrins.get(str_op,  mode='w'))

    ko1, ki1 = s[step2_buf].split(step2_buf.op.reduce_axis[0], factor=1)
    xo1,xi1 = s[step2_buf].split(step2_buf.op.axis[2], factor=16)
    s[step2_buf].reorder( step2_buf.op.axis[0],step2_buf.op.axis[1],xo1,ko1,ki1, xi1)
    s[step2_buf].tensorize(ki1, env.intrins.get(str_op,  mode='w',nDim=3))
    
    print(nnpu.lower(s, [a, step2_host], simple_mode=True))
   # exit()
    func = nnpu.build(s, [a, step2_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=in_shape, dtype=a.dtype, low = -128, high = 127)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros(out_shape, dtype=step2_host.dtype), ctx)

    func(a_nd, c_nd)
    print("pooling-max")
    print(c_nd.asnumpy())
    
    print("nppooling-max")
    gt=max_pooling(in_shape,out_shape,cell_shape,a_np,a.dtype)
    print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)
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