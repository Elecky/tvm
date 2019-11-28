import tvm
import time
import numpy as np
import numpy
from tvm import autotvm
from tvm import relay
from tvm.relay.testing import check_grad, ctx_list, run_infer_type
from tvm.relay.transform import gradient
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import topi
import topi.testing
import logging
import sys
import os

device = "cuda"

log_file = "cuda_vectoradd.log"
dtype = 'float32'

ctx = tvm.context(device, 0)

tuning_option = {
    'log_filename': log_file,

    'tuner': 'ga',
    'n_trial': 200,
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
    builder=autotvm.LocalBuilder(timeout=10),
    runner=autotvm.LocalRunner(number=20,repeat=3,timeout=4,min_repeat_ms=150),
    # runner=autotvm.RPCRunner(
    #     'titanv100',  # change the device key to your key
    #     '0.0.0.0', 9190,
    #     number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}



@tvm.register_func("tvm.contrib.my_tvm_poolgrad")
def my_tvm_poolgrad(out_grad_np,a_np):
    """pool_grad for NCHW layout in python"""
    pool_size=(1,1)
    strides=(1,1)
    padding=(0,0)
    pool_type='max'
    ceil_mode=False
    count_include_pad=True
    dtype = a_np.dtype
    n, ic, ih, iw = a_np.shape
    kh, kw = pool_size
    sh, sw = strides
    pt, pl, pb, pr = padding

    pad_np = np.zeros(shape=(n, ic, ih+pt+pb, iw+pl+pr)).astype(dtype)
    no_zero = (range(n), range(ic), (range(pt, ih+pt)), (range(pl, iw+pl)))
    pad_np[np.ix_(*no_zero)] = a_np
    _, _, oh, ow = out_grad_np.shape
    pool_grad_np = np.zeros(shape=a_np.shape)
    pad_pool_grad_np = np.zeros(shape=pad_np.shape)

    if pool_type == 'avg':
        for i in range(oh):
            for j in range(ow):
                if count_include_pad:
                    shape = pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw].shape
                    # this can be different from kh*kw if input size cannot divide stride
                    pad_count = shape[2] * shape[3]
                else:
                    pad_count = np.sum(
                        pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] > 0, axis=(2, 3))
                    # take the first element, as they are the same across batch and channel
                    pad_count = pad_count.ravel()[0]
                pad_pool_grad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] += \
                        out_grad_np[:, :, i, j].reshape(n, ic, 1, 1) / np.maximum(pad_count, 1)
    elif pool_type == 'max':
        start=time.time()
        for i in range(oh):
            for j in range(ow):
                a_patch = pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
                a_patch = np.reshape(a_patch, (n, ic, -1))
                #得到最大值索引
                max_indices = np.argmax(a_patch, axis=2)
                c_idx, n_idx = np.meshgrid(range(ic), range(n), sparse=True)
                #得到原数组中的索引
                h_idx, w_idx = np.unravel_index(max_indices, (kh, kw))
                pad_pool_grad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw][n_idx, c_idx, h_idx, w_idx] += \
                    out_grad_np[n_idx, c_idx, i, j]
        end=time.time()
        print("算子运行时间：")
        print(end-start)
    for i in range(pool_grad_np.shape[2]):
        for j in range(pool_grad_np.shape[3]):
            pool_grad_np[:, :, i, j] = pad_pool_grad_np[:, :, i + pt, j + pl]

    return pool_grad_np

def maxpoolgrad_naive(out_grad,data):
    A=tvm.placeholder(out_grad.shape,name='A',dtype=dtype)
    B=tvm.placeholder(data.shape,name='B',dtype=dtype)
    C=tvm.extern(A.shape,[A,B],lambda ins,outs:tvm.call_packed(
        "tvm.contrib.my_tvm_poolgrad",
        ins[0],ins[1],outs[0]),name="C")
    s=tvm.create_schedule(C.op)
    #此处可做schedule?


    print(C.op)
    #bx, tx = s[C].split (C.op.axis[0], factor=64)
    #s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    #s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    module=tvm.build(s,[A,B,C],device,target_host="llvm")
    #evaluator=module.time_evaluator(module.entry_name,ctx,number=100)
    a = tvm.nd.array(out_grad, ctx)
    b = tvm.nd.array(data, ctx)
    c = tvm.nd.array(np.zeros(data.shape), ctx)
    module(a,b,c)

    print(c)
def verify_max_pool2d_grad(x_shape, pool_size, strides, padding, ceil_mode):
    x = relay.var("x", relay.TensorType(x_shape, "float32"))
    y = tvm.relay.nn.max_pool2d(x, pool_size=pool_size, strides=strides, padding=padding,
                                ceil_mode=ceil_mode)

    fwd_func = relay.Function([x], y)
    fwd_func = run_infer_type(fwd_func)
    bwd_func = run_infer_type(gradient(fwd_func))

    data = np.random.rand(*x_shape).astype("float32")
    ph, pw = padding
    y_shape = topi.util.get_const_tuple(fwd_func.ret_type.shape)
    out_grad = np.ones(shape=y_shape)
    maxpoolgrad_naive(out_grad,data)

verify_max_pool2d_grad((1, 4, 16, 16), pool_size=(2, 2), strides=(2, 2), padding=(0, 0),
                           ceil_mode=False)
