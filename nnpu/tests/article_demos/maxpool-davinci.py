import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='test of NNPU Op')
parser.add_argument('--sim', type=str, help='the simulator to use', 
                    default='S0', choices=['S0', 'S1', 'SC'])
args = parser.parse_args()

factor = 64
dim_x, dim_y, dim_c = 2, 4, 256
cfg_path = './nnpu_config.davinci.yaml'

def max_pooling(inshape,outshape,cell_shape,innp,outdetype):
  ret=np.full(outshape, np.iinfo(outdetype).min , dtype=outdetype)
  for w in range(outshape[0]):
    for h in range(outshape[1]):
      for j in range(cell_shape):
        for k in range(cell_shape):
          for l in range(outshape[2]):
            ret[w][h][l]=max(ret[w][h][l],innp[w*cell_shape+j][h*cell_shape+k][l])
  return ret

# reduce max
with nnpu.Environment(cfg_path):
    env = nnpu.get_env()
    nnpu.set_device(env, type=args.sim)
    nnpu.set_dump(False)

    #==================================#
    # ------ first define shapes ------
    #==================================#
    
    # input data layout: HWC
    in_shape = (40, 40, 256)
    # pooling windows size, height == width.
    cell_shape = 2
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
    data = tvm.placeholder(in_shape, dtype_n, 'a')

    # stage 1, find the maximum pixel in every pooling window.
    # the extent of two reduction axes are sizes of pooling window.
    k1 = tvm.reduce_axis((0,cell_shape), 'k1')
    k2 = tvm.reduce_axis((0,cell_shape), 'k2')
    pooling_buf = tvm.compute(out_shape, 
                        lambda i,j,k: 
                         tvm.max(data[i * cell_shape + k1, j * cell_shape + k2, k],
                                 axis=[k1, k2]),
                       'pooling_buf')
    pooling_host = tvm.compute(out_shape, lambda *i: pooling_buf(*i), 'pooling_host')

    # ------ this ends the computation description. ------

    #==================================#
    # ------ begin scheduling ------
    #==================================#
    s = tvm.create_schedule(pooling_host.op)    
    
    data_l1 = s.cache_read(data, env.get_scope('buffer0'), pooling_buf)
    data_uni = s.cache_read(data_l1, env.get_scope('buffer4'), pooling_buf)
    pooling_l1 = s.cache_write(pooling_host, env.get_scope('buffer0'))

    # set scopes
    s[pooling_buf].set_scope(env.get_scope('buffer4'))

    #tensorize
    i, j, c = pooling_buf.op.axis
    k1, k2 = pooling_buf.op.reduce_axis
    # split the reduce_axis by factor 1, to produce a dummy reduce axis. 
    # this is a trick to enable tensorize, due to limitation of tvm's tensorize pattern matcher.
    k2, ki = s[pooling_buf].split(k2, factor=1)
    co, ci = s[pooling_buf].split(c, factor=factor)
    # reorder axes.
    # put j right before j to eliminate memory dependency between two consecutive VGTMV instruction
    s[pooling_buf].reorder(co, k1, k2, i, j, ki, ci)
    s[pooling_buf].tensorize(ki, env.intrins.get(str_op, scope_out='buffer4', scope_in='buffer4', mode='n', size=factor))
    # attack data load into pooling stage
    s[data_l1].compute_at(s[pooling_buf], co)
    s[data_uni].compute_at(s[pooling_buf], co)

    # output blocking
    x, y, c = s[pooling_host].op.axis
    xo, yo, xi, yi = s[pooling_host].tile(x, y, dim_x, dim_y)
    x_vt, xo = s[pooling_host].split(xo, nparts=2)
    co, ci = s[pooling_host].split(c, dim_c)
    s[pooling_host].reorder(x_vt, xo, yo, co, xi, yi, ci)
    s[pooling_buf].compute_at(s[pooling_host], co)
    s[pooling_l1].compute_at(s[pooling_host], co)
    s[pooling_host].bind(x_vt, tvm.thread_axis('cthread'))

    # pragma
    s[data_l1].pragma(data_l1.op.axis[0], env.dma_copy_to_buf)
    s[data_uni].pragma(data_uni.op.axis[0], env.scratchpad_copy)
    s[pooling_l1].pragma(pooling_l1.op.axis[0], env.scratchpad_copy)
    s[pooling_host].pragma(xi, env.dma_copy_from_buf)
    #==================================#
    # ------ this ends the scheduling ------
    #==================================#
    # print(tvm.lower(s, [data, pooling_host], simple_mode=True))
    print(nnpu.lower(s, [data, pooling_host], simple_mode=True))
    # exit()
    func = nnpu.build(s, [data, pooling_host], 'nnpu', 'llvm', name='nnpu_func')

    ctx = tvm.nd.TVMContext(13, 0)
    data_np = np.random.randint(size=in_shape, dtype=data.dtype, low = -128, high = 127)
    data_nd = tvm.nd.array(data_np, ctx)

    res_nd = tvm.nd.array(np.zeros(out_shape, dtype=pooling_host.dtype), ctx)

    func(data_nd, res_nd)

    gt=max_pooling(in_shape,out_shape,cell_shape,data_np,data.dtype)
    # print(gt)
    np.testing.assert_allclose(res_nd.asnumpy(), gt)
    print('test passed')