import tvm
import numpy as np

h, w, c = 128, 128, 32
shape = (h, w, c)
feature = tvm.placeholder(shape, 'float32', 'feature')
roi = tvm.placeholder((4, ), 'int32', 'roi')

kh = tvm.reduce_axis((roi[0], roi[1]), 'kh')
kw = tvm.reduce_axis((roi[2], roi[3]), 'kw')

res = tvm.compute((32, ), lambda i: tvm.sum(feature[kh, kw, i], axis=[kh, kw]), 'res')

s = tvm.create_schedule(res.op)

print(tvm.lower(s, [feature, roi, res], simple_mode=True))
func = tvm.build(s, [feature, roi, res], 'llvm', 'llvm', 'func')

f_np = np.random.random(shape).astype('float32')
f_nd = tvm.nd.array(f_np)
roi_np = np.zeros((4,), dtype='int32')
roi_np[0] = 24
roi_np[1] = 36
roi_np[2] = 44
roi_np[3] = 97
roi_nd = tvm.nd.array(roi_np)

res = tvm.nd.array(np.zeros((32, ), dtype='float32'))

func(f_nd, roi_nd, res)

print(res.asnumpy())