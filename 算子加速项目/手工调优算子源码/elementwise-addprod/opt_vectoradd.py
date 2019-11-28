import tvm
import time
import numpy as np
import numpy
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import logging
import sys
import os
import topi


device = "cuda"
dtype = 'float32'
ctx = tvm.context(device, 0)

def vectoradd_naive():
    N = 1048576
    A = tvm.placeholder ((N,), name='A', dtype=dtype)
    B = tvm.placeholder ((N,), name='B', dtype=dtype)
    a=2
    C = tvm.compute (A.shape, lambda *i: A(*i) +B(*i), name='C')
    s = tvm.create_schedule (C.op)

    bx, tx = s[C].split (C.op.axis[0], factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

    module = tvm.build(s, [A, B, C], device, target_host="llvm")

    print(tvm.lower(s, [A, B, C], simple_mode=True))

    a = numpy.random.rand(N).astype(dtype)
    a_np = tvm.nd.array(a, ctx)
    b = numpy.random.rand(N).astype(dtype)
    b_np = tvm.nd.array(b, ctx)
    #c_np = np.multiply(a,b)

    c_tvm = tvm.nd.array(numpy.random.rand(N).astype(dtype), ctx)
    module(a_np, b_np, c_tvm)
    
   # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    evaluator = module.time_evaluator(module.entry_name, ctx, number=100)
    print('Naive: %f ms' % (evaluator(a_np, b_np, c_tvm).mean*1e3))

vectoradd_naive()