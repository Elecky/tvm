from __future__ import absolute_import, print_function
# import nnvm
import numpy as np 
import nnvm.compiler
from tvm.contrib import graph_runtime as runtime  
import tvm
import nnpu
from nnpu.utils import ScheduleProcHelper

from nnvm.testing import utils
import nnvm.testing

import logging
from collections import namedtuple
import time
logging.basicConfig()

def test_relu():
    shape = (2, 16)
    inputs = nnvm.symbol.Variable("inputs")
    env = nnpu.get_env()
    target_host = "llvm"
    device = "nnpu"
    target = tvm.target.create("llvm -device={}".format(device))
    z = nnvm.symbol.relu(inputs)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(),type='S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random(size = (2, 16)).astype("float32") - 0.5
        m.set_input(**{'inputs':a_np})
        m.run()
        out = m.get_output(0, out = tvm.nd.empty((2, 16)))
        print(a_np)
        print(out.dtype)
        print(out)
        np.testing.assert_allclose(out.asnumpy(), np.maximum(a_np, 0))
        print("tests")
        print(compute_graph.ir())
        print(deploy_graph.ir())
def test_tanh():
    shape = (2, 16)
    inputs = nnvm.symbol.Variable("inputs")
    target_host = "llvm"
    device = "nnpu"
    env = nnpu.get_env()
    target = tvm.target.create("llvm -device={}".format(device))
    z = nnvm.symbol.tanh(inputs)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs" : shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random(size = (2, 16))
        # gt = np.exp(a_np) / (1 + np.exp(a_np))
        m.set_input(**{'inputs':a_np})
        m.run()
        out = m.get_output(0, out = tvm.nd.empty((2, 16)))
        print(out)
        # np.testing.assert_allclose(out.asnumpy(), gt)
        # print("tests")
        # print(compute_graph.ir())
        print(deploy_graph.ir())

def test_sigmoid():
    shape = (2, 16)
    inputs = nnvm.symbol.Variable("inputs")
    env = nnpu.get_env()
    target_host = "llvm"
    device = "nnpu"
    target = tvm.target.create("llvm -device={}".format(device))
    z = nnvm.symbol.sigmoid(inputs)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs" : shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random(size = (2, 16))
        gt = np.exp(a_np) / (1 + np.exp(a_np))
        m.set_input(**{'inputs':a_np})
        m.run()
        out = m.get_output(0, out = tvm.nd.empty((2, 16)))
        print(out)
        np.testing.assert_allclose(out.asnumpy(), gt)
        print("tests")
        print(compute_graph.ir())
        print(deploy_graph.ir())
def test_dense():
    shape = (1, 802816)
    weight_shape = (32, 802816)
    bias_shape = (32,)
    inputs = nnvm.symbol.Variable("inputs")
    weights = nnvm.symbol.Variable("weights")
    bias = nnvm.symbol.Variable("bias")
    env = nnpu.get_env()
    target_host = "llvm"
    device = "nnpu"
    target = tvm.target.create("llvm -device={}".format(device))
    z = nnvm.symbol.dense(data = inputs, weight = weights, bias = bias, units = 32)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape, "weights":weight_shape, "bias":bias_shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape, "weights":weight_shape, "bias":bias_shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random(size = shape)
        b_np = np.random.random(size = weight_shape)

        c_np = np.random.random(size = bias_shape)

        m.set_input(**{"inputs":a_np, "weights":b_np, "bias":c_np})
        m.run()
        gt = a_np.dot(b_np.transpose()) + c_np

        out = m.get_output(0, out = tvm.nd.empty((1, 32)))
        np.testing.assert_allclose(out.asnumpy(), gt, rtol = 5e-5)
        print("tests")
        print(out)
        print(compute_graph.ir())
        print(deploy_graph.ir())

def max_pooling(inshape, outshape, cell_shape, innp, stride, outdtype):
    ret=np.zeros(outshape, dtype=outdtype)
    for b in range(outshape[0]):
        for w in range(outshape[1]):
            for h in range(outshape[2]):
                for l in range(outshape[3]):
                    for j in range(cell_shape[0]):
                        for k in range(cell_shape[1]):
                            ret[b][w][h][l] = max(ret[b][w][h][l], innp[b][w*stride[0] + j][h*stride[1] + k][l])
    return ret
    
def avg_pooling(inshape, outshape, cell_shape, innp, stride, outdtype):
    ret=np.zeros(outshape, dtype=outdtype)
    for b in range(outshape[0]):
        for w in range(outshape[1]):
            for h in range(outshape[2]):
                for l in range(outshape[3]):
                    for j in range(cell_shape[0]):
                        for k in range(cell_shape[1]):
                            ret[b][w][h][l] = ret[b][w][h][l] + innp[b][w*stride[0] + j][h*stride[1] + k][l]
                    ret[b][w][h][l] = ret[b][w][h][l] / (cell_shape[0] * cell_shape[1])
    return ret

def test_max_pool2d():
      
    device = "nnpu"
    target = tvm.target.create("llvm -device={}".format(device))
    target_host = "llvm"
    inputs = nnvm.symbol.Variable("inputs")
    shape = (1, 224, 224, 16)
    kernels = nnvm.symbol.Variable("kernels")
    kernel_shape = (2, 2)
    z = nnvm.symbol.avg_pool2d(inputs, pool_size = (2, 2), strides = (1, 1), layout = "NHWC")
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32")
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32")
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random(size = (1, 224, 224, 16))
        m.set_input(**{"inputs":a_np})
        m.run()
        
        out = m.get_output(0, out = tvm.nd.empty((1, 223, 223, 16)))
        gt = avg_pooling((1, 224, 224, 16), (1, 223, 223, 16), (2, 2), a_np, (1, 1), "float32")
        np.testing.assert_allclose(out.asnumpy(), gt, rtol=5e-7)
        print("max_pool2d tests success")
        print(gt)
        print(out)
        print("end")
def test_global_max_pool2d():
    device = "nnpu"
    target = tvm.target.create("llvm -device={}".format(device))
    target_host = "llvm"
    inputs = nnvm.symbol.Variable("inputs")
    shape = (1, 224, 224, 16)
    kernels = nnvm.symbol.Variable("kernels")
    kernel_shape = (2, 2)
    z = nnvm.symbol.global_max_pool2d(inputs, layout = "NHWC")
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32")
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32")
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random(size = (1, 224, 224, 16))
        print(a_np)
        m.set_input(**{"inputs":a_np})
        m.run()
        
        out = m.get_output(0, out = tvm.nd.empty((1, 1, 1, 16)))
        gt = max_pooling((1, 224, 224, 16), (1, 1, 1, 16), (224, 224), a_np, (1, 1), "float32")
        np.testing.assert_allclose(out.asnumpy(), gt, rtol=5e-07)
        print("global_max_pool2d tests success")
        print(gt)
        print(out)
        print("end")
def test_elemwise_add():
    env = nnpu.get_env()
    device = "nnpu"
    target_host = "llvm"
    target = tvm.target.create("llvm -device={}".format(device))
    inputs1 = nnvm.symbol.Variable("inputs1")
    inputs2 = nnvm.symbol.Variable("inputs2")
    shape = (1116, 16)
    z = nnvm.symbol.elemwise_add(inputs1, inputs2)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs1":shape, "inputs2":shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs1":shape, "inputs2":shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random((1116, 16))
        b_np = np.random.random((1116, 16))
        print("a_np : ")
        print(a_np)
        print("b_np : ")
        print(b_np)
        m.set_input(**{"inputs1":a_np, "inputs2":b_np})
        m.run()
        gt = a_np + b_np
        out = m.get_output(0, out = tvm.nd.empty((1116, 16)))
        np.testing.assert_allclose(out.asnumpy(), gt)
        print(out)
        print("elemwise_add tests success")

def test_elemwise_sub():
    env = nnpu.get_env()
    device = "nnpu"
    target_host = "llvm"
    target = tvm.target.create("llvm -device={}".format(device))
    inputs1 = nnvm.symbol.Variable("inputs1")
    inputs2 = nnvm.symbol.Variable("inputs2")
    shape = (16, 6)
    z = nnvm.symbol.elemwise_sub(inputs1, inputs2)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs1":shape, "inputs2":shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs1":shape, "inputs2":shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random((16, 6))
        b_np = np.random.random((16, 6))
        print("a_np : ")
        print(a_np)
        print("b_np : ")
        print(b_np)
        m.set_input(**{"inputs1":a_np, "inputs2":b_np})
        m.run()
        gt = (a_np.astype("float32") - b_np.astype("float32")).astype("float32")
        out = m.get_output(0, out = tvm.nd.empty((16, 6)))
        np.testing.assert_allclose(out.asnumpy(), gt)
        print(out)
        print("elemwise_sub tests success")
def test_elemwise_mul():
    env = nnpu.get_env()
    device = "nnpu"
    target_host = "llvm"
    target = tvm.target.create("llvm -device={}".format(device))
    inputs1 = nnvm.symbol.Variable("inputs1")
    inputs2 = nnvm.symbol.Variable("inputs2")
    shape = (16, 6, 16)
    z = nnvm.symbol.elemwise_mul(inputs1, inputs2)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs1":shape, "inputs2":shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs1":shape, "inputs2":shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random((16, 6, 16))
        b_np = np.random.random((16, 6, 16))
        print("a_np : ")
        print(a_np)
        print("b_np : ")
        print(b_np)
        m.set_input(**{"inputs1":a_np, "inputs2":b_np})
        gt = (a_np.astype("float32") * b_np.astype("float32")).astype("float32")
        m.run()
        out = m.get_output(0, out = tvm.nd.empty((16, 6, 16)))
        np.testing.assert_allclose(out.asnumpy(), gt)
        print("elemwise_mul tests success")
        print(out)
def test_onemore():
    shape = (1, 32, 32, 16)
    inputs = nnvm.symbol.Variable("inputs")
    env = nnpu.get_env()
    target_host = "llvm"
    device = "nnpu"
    target = tvm.target.create("llvm -device={}".format(device))
    
    z1 = nnvm.symbol.relu(inputs)
    z = nnvm.symbol.sqrt(z1)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(),type='S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random(size = (1, 32, 32, 16))
        m.set_input(**{'inputs':a_np})
        m.run()
        out = m.get_output(0, out = tvm.nd.empty((1, 32, 32, 16)))
        print(out)
        print(compute_graph.ir())
        print(deploy_graph.ir())
def test_softmax():
    env = nnpu.get_env()
    shape = (1122, 16)
    device = "nnpu"
    target_host = "llvm"
    target = tvm.target.create("llvm -device={}".format(device))
    input = nnvm.symbol.Variable("input")
    z = nnvm.symbol.softmax(input)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"input":shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"input":shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random((1122, 16))
        print(a_np)
        m.set_input(**{"input":a_np})
        m.run()
        out = m.get_output(0, out = tvm.nd.empty((1122, 16)))
        s = np.exp(a_np.astype("float32"))
        gt = (s.astype("float32") / (s.sum(axis = 1).reshape(1122, 1))).astype("float32")
        np.testing.assert_allclose(out.asnumpy(), gt, rtol=5e-7)
        print("softmax tests success")
        print(out)
        print(compute_graph.ir())
        print(deploy_graph.ir())

def test_exp():
    env = nnpu.get_env()
    shape = (2, 16)
    device = "nnpu"
    target_host = "llvm"
    target = tvm.target.create("llvm -device={}".format(device))
    inputs = nnvm.symbol.Variable("inputs")
    z = nnvm.symbol.exp(inputs)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs" : shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs" : shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random((2, 16))
        print(a_np)
        m.set_input(**{"inputs" : a_np})
        m.run()
        gt = np.exp(a_np.astype("float32")).astype("float32")
        out = m.get_output(0, out = tvm.nd.empty((2, 16)))
        np.testing.assert_allclose(out.asnumpy(), gt)
        print("exp tests success")
        print(out)
        print(compute_graph.ir())
        print(deploy_graph.ir())

def test_log():
    env = nnpu.get_env()
    shape = (1, 22, 22, 16)
    device = "nnpu"
    target_host = "llvm"
    target = tvm.target.create("llvm -device={}".format(device))
    inputs = nnvm.symbol.Variable("inputs")
    z = nnvm.symbol.log(inputs)
    z1 = nnvm.symbol.exp(z)
    compute_graph = nnvm.graph.create(z1)
    with nnvm.compiler.build_config(opt_level = 1):
        if target.device_name != "nnpu":
            deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs" : shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs" : shape}, dtype = "float32", target_host = target_host)
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        m = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.random(shape)
        print(a_np)
        m.set_input(**{"inputs" : a_np})
        m.run()
        out = m.get_output(0, out = tvm.nd.empty(shape))
        gt = np.exp(np.log(a_np.astype("float32")).astype("float32")).astype("float32")
        print(out)
        np.testing.assert_allclose(out.asnumpy(), gt)
        print("log tests success")
        print(compute_graph.ir())
        print(deploy_graph.ir())
        
def test_conv2d():
    input_shape = (1, 16, 9, 64)
    target_host = "llvm"
    device = "nnpu"
    target = tvm.target.create("llvm -device={}".format(device))
    inputs = nnvm.symbol.Variable("inputs")
    z1 = nnvm.symbol.conv2d(data = inputs, channels = 64, kernel_size=(3, 3), padding = (0, 0), use_bias=False,
                                layout='NHWC', kernel_layout='HWOI')
    z = nnvm.symbol.relu(z1)

    compute_graph = nnvm.graph.create(z)
        
    with nnvm.compiler.build_config(opt_level = 1):
        if target.device_name != "nnpu":

                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs" : input_shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'SC')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs" : input_shape}, dtype = "float32", target_host = target_host)

        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        module = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.uniform(size  = input_shape, low = -32, high = 32).astype(np.float32)
 
        module.set_input(inputs = a_np)
        module.run()
        print(deploy_graph.ir())
        out = module.get_output(0, out = tvm.nd.empty((1, 14, 7, 64)))
        
"""

nnpu.set_dump(False)
print("test_relu           :      ")
test_relu()
print("test_dense          :      ")
# test_dense()
print("test_max_pool2d     :      ")
test_max_pool2d()
print("test_elemwise_add   :      ")
test_elemwise_add()
print("test_elemwise_sub   :      ")
test_elemwise_sub()
print("test_elemwise_mul   :      ")


test_global_max_pool2d()
# print("test_softmax        :      ")
# test_softmax()
print("test_sigmoid        :      ")
test_sigmoid()
print("test_exp            :      ")
test_exp()
print("test_log            :      ")
test_log()
print("test_conv2d         :      ")
test_conv2d()
"""
nnpu.set_dump(False)
test_conv2d()