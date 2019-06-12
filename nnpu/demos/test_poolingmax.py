import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

# reduce max
def test():
    env = nnpu.get_env()
    nnpu.set_dump(False)

    #==================================#
    # ------ first define shapes ------
    #==================================#
    
    # input data layout: HWC
    in_shape = (32, 32, 128)
    # pooling windows size, height == width.
    cell_shape = 4
    # in this demo we don't do padding, so input data height and width must be divisible to pooling window size.
    assert in_shape[0] % cell_shape == 0, 'error'
    assert in_shape[1] % cell_shape == 0, 'error'
    nvctr_unit = env.cfg['vector_unit']['size']
    assert in_shape[2] % nvctr_unit == 0, 'channel not divisible to vector unit size'

    out_shape = (in_shape[0] // cell_shape,in_shape[1] // cell_shape,in_shape[2])
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    
    sph = ScheduleProcHelper()
    str_op = 'VGTMMerge'

    #=================================================================#
    # ------ after all shapes defined, begin compute describing. ------
    #=================================================================#
    a = tvm.placeholder(in_shape, dtype_w, 'a')
    # first copy to scratchpad.
    a_buf, _1 = nnpu.utils.CopyHtoBuf(a, 'a', sph)

    # stage 1, find the maximum pixel in every pooling window.
    # the extent of two reduction axes are sizes of pooling window.
    k1 = tvm.reduce_axis((0,cell_shape), 'k1')
    k2 = tvm.reduce_axis((0,cell_shape), 'k2')
    pooling_buf = tvm.compute(out_shape, 
                        lambda i,j,k: 
                         tvm.max(a_buf[i * cell_shape + k1, j * cell_shape + k2, k],
                                 axis=[k1, k2]),
                       'pooling_buf')
    sph.MarkScope(pooling_buf, 'buffer0')
    
    # copy back to host.    
    step2_host, step2_dram = nnpu.utils.CopyBufToH(pooling_buf, 'pooling',sph)
    # ------ this ends the computation description. ------

    #==================================#
    # ------ begin scheduling ------
    #==================================#
    s = tvm.create_schedule(step2_host.op)    
    sph.Transform(s)

    #tensorize
    i, j, k = pooling_buf.op.axis
    k1, k2 = pooling_buf.op.reduce_axis
    # split the reduce_axis by factor 1, to produce a dummy reduce axis. 
    # this is a trick to enable tensorize, due to limitation of tvm's tensorize pattern matcher.
    ko, ki = s[pooling_buf].split(k2, factor=1)
    xo, xi = s[pooling_buf].split(k, factor=16)
    # reorder axes.
    # put xo right before ki to eliminate memory dependency between two consecutive VGTMV instruction
    s[pooling_buf].reorder( i, j, k1, ko, xo, ki, xi)
    s[pooling_buf].tensorize(ki, env.intrins.get(str_op, scope_out='buffer0', mode='w'))
    # unroll
    # s[pooling_buf].unroll(ko)
    # s[pooling_buf].unroll(xo)
    #==================================#
    # ------ this ends the scheduling ------
    #==================================#

    print(nnpu.lower(s, [a, step2_host], simple_mode=True))
    # exit()
    func = nnpu.build(s, [a, step2_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=in_shape, dtype=a.dtype, low = -128, high = 127)
    a_nd = tvm.nd.array(a_np, ctx)

    c_nd = tvm.nd.array(np.zeros(out_shape, dtype=step2_host.dtype), ctx)

    func(a_nd, c_nd)
    # print("pooling-max")
    # print(c_nd.asnumpy())
    
    # print("nppooling-max")
    gt=max_pooling(in_shape,out_shape,cell_shape,a_np,a.dtype)
    # print(gt)
    np.testing.assert_allclose(c_nd.asnumpy(), gt)
    print('test passed')
    
def max_pooling(inshape,outshape,cell_shape,innp,outdetype):
  ret=np.zeros(outshape, dtype=outdetype)
  for w in range(outshape[0]):
    for h in range(outshape[1]):
      for j in range(cell_shape):
        for k in range(cell_shape):
          for l in range(outshape[2]):
            ret[w][h][l]=max(ret[w][h][l],innp[w*cell_shape+j][h*cell_shape+k][l])
  return ret

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test of NNPU Op')
    parser.add_argument('--sim', type=str, help='the simulator to use', 
                        default='S0', choices=['S0', 'S1', 'SC'])
    args = parser.parse_args()

    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    test()