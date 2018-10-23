import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

env = nnpu.get_env()
nnpu.set_device(env)

out_channel = 32
in_channel = 64
batch = 1
gemm_shape = (16, 16, 1)  # the shape of gemm instruction

assert out_channel % gemm_shape[0] == 0, 'out_channel not divisble to gemm insn input1 row count'
assert in_channel % gemm_shape[1] == 0, 'in_channel not divisble to gemm insn factor'

weight_shape = (out_channel // gemm_shape[0], in_channel // gemm_shape[1], 
                gemm_shape[0], gemm_shape[1])
data_shape = (in_channel // gemm_shape[1], gemm_shape[1])
prod_shape = (weight_shape[0], weight_shape[1], weight_shape[2])
mm_shape = (prod_shape[0], prod_shape[2])

dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']

weights = tvm.placeholder(weight_shape, dtype_n, 'weights')
data = tvm.placeholder(data_shape, dtype_n, 'data')
bias = tvm.placeholder(mm_shape, dtype_w, 'bias')

sph = ScheduleProcHelper()

# dma copy to device
w_buf, w_dram = nnpu.utils.CopyHtoBuf(weights, 'weight', sph)
d_buf, d_dram = nnpu.utils.CopyHtoBuf(data, 'data', sph)
bias_buf, bias_dram = nnpu.utils.CopyHtoBuf(bias, 'bias', sph)

# gemm
k = tvm.reduce_axis((0, gemm_shape[1]), 'k0')
prod_buf = tvm.compute(prod_shape, 
                       lambda i, j, l: 
                        tvm.sum(w_buf[i, j, l, k].astype(dtype_w) * d_buf[j, k].astype(dtype_w), 
                                axis = k),
                       'prod_buf')
sph.MarkScope(prod_buf)

# reduce along in_channel tiles
k = tvm.reduce_axis((0, prod_shape[1]), 'k1')
mm_buf = tvm.compute(mm_shape, lambda i, j: tvm.sum(prod_buf[i, k, j], axis=k), 'mm_buf')
sph.MarkScope(mm_buf)

intern_buf = tvm.compute(mm_shape, lambda *i: mm_buf(*i) + bias_buf(*i), 'intern_buf')
sph.MarkScope(intern_buf)

zeroImm = tvm.const(0).astype(dtype_w)
out_buf = tvm.compute(mm_shape, 
                      lambda *i: tvm.select(intern_buf(*i) > zeroImm, intern_buf(*i), zeroImm),
                      'out')
sph.MarkScope(out_buf)
out_host, _ = nnpu.utils.CopyBufToH(out_buf, 'out', sph)

# transform schedule
s = tvm.create_schedule(out_host.op)
sph.Transform(s)

s[prod_buf].tensorize(prod_buf.op.axis[2], 
                      env.intrins.get('GEMM', shape=gemm_shape, reduce=True, mode='inc'))

ko, ki = s[mm_buf].split(s[mm_buf].op.reduce_axis[0], factor=1)
s[mm_buf].reorder(mm_buf.op.axis[0], ko, ki, mm_buf.op.axis[1])
s[mm_buf].tensorize(ki, env.intrins.get('VAddMerge', mode='w'))

s[intern_buf].tensorize(intern_buf.op.axis[1], env.intrins.get('VAddV', mode='w'))

s[out_buf].tensorize(out_buf.op.axis[1], env.intrins.get('VGTMI', mode='w', imm_value=0))

# compute at
s[prod_buf].compute_at(s[mm_buf], ko)

# build
print(nnpu.lower(s, [weights, data, bias,out_host], simple_mode=True))

func = nnpu.build(s, [weights, data, bias, out_host], 'nnpu', 'llvm', name='nnpu_func')

# run
ctx = tvm.nd.TVMContext(13, 0)

weight_np = np.random.randint(-32, 32, size=(out_channel, in_channel), dtype=weights.dtype)
w_np = np.reshape(weight_np, (weight_shape[0], weight_shape[2], weight_shape[1], weight_shape[3]))
w_np = np.transpose(w_np, axes=(0, 2, 1, 3))
w_nd = tvm.nd.array(w_np, ctx)

data_np = np.random.randint(-32, 32, size=(in_channel,), dtype=data.dtype)
d_np = np.reshape(data_np, (data_shape[0], data_shape[1]))
d_nd = tvm.nd.array(d_np, ctx)

bias_np = np.random.randint(-1000, 1000, size=(out_channel), dtype=bias.dtype)
b_np = np.reshape(bias_np, mm_shape)
b_nd = tvm.nd.array(b_np, ctx)

mm_nd = tvm.nd.array(np.zeros(mm_shape, dtype=out_host.dtype), ctx)

func(w_nd, d_nd, b_nd, mm_nd)

gt = np.dot(weight_np.astype(dtype_w), data_np)
gt = gt + bias_np
print(gt)
print(np.reshape(mm_nd.asnumpy(), (-1, )))