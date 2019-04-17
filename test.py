from __future__ import absolute_import, print_function
import nnvm
import numpy as np 

import nnvm.compiler
from tvm.contrib import graph_runtime as runtime
from nnvm.testing import utils
import nnvm.testing
import tvm
import logging
from collections import namedtuple
import time
logging.basicConfig()


def test_conv2d():
        input_shape = (1, 224, 224, 32)
        target = tvm.target.create("llvm")
        inputs = nnvm.symbol.Variable("inputs")
        z = nnvm.symbol.conv2d(data=inputs, channels = 32, kernel_size=(7, 7), strides = (2, 2), padding = (3, 3), use_bias=False,
                                layout='NHWC', kernel_layout='HWOI')
        compute_graph = nnvm.graph.create(z)
        
        with nnvm.compiler.build_config(opt_level=0):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs" : input_shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                module = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random((1, 224, 224, 32))
                print(a_np)
                module.set_input(inputs = a_np)
                module.run()
                out = module.get_output(0, out = tvm.nd.empty((1, 112, 112, 32)))
                print(out.asnumpy)
                print(compute_graph.ir())
                print(deploy_graph.ir())

def test_dense():
        input_shape = (1, 8028160)
        weight_shape = (32, 8028160)
        inputs = nnvm.symbol.Variable("inputs")
        weights = nnvm.symbol.Variable("weights")
        target = tvm.target.create("llvm")
        z = nnvm.symbol.dense(inputs, weights, units = 32, use_bias = False)
        compute_graph = nnvm.graph.create(z)
        with nnvm.compiler.build_config(opt_level = 0):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs" : input_shape, "weights" : weight_shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                m = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random(size = (1, 8028160))
                b_np = np.random.random(size = (32, 8028160))
                print(a_np)
                
                m.set_input(**{'inputs': a_np, 'weights': b_np})
                m.run()
                out = m.get_output(0, out = tvm.nd.empty((1, 32), ctx = ctx))
                print(out.asnumpy)
                print(compute_graph.ir())
                print(deploy_graph.ir())
def test_maxpool():
        input_shape = (1, 224, 224, 32)
        target = tvm.target.create("llvm")
        inputs = nnvm.symbol.Variable("inputs")
        z = nnvm.symbol.max_pool2d(data=inputs, pool_size=(7, 7), layout = 'NHWC')
        compute_graph = nnvm.graph.create(z)
        
        with nnvm.compiler.build_config(opt_level=0):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs" : input_shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                module = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random(input_shape)
                print(a_np)
                module.set_input(inputs = a_np)
                module.run()
                out = module.get_output(0, out = tvm.nd.empty((1, 218, 218, 32)))
                print(out.asnumpy)
                print(compute_graph.ir())
                print(deploy_graph.ir())
def test_softmax():
        inputs = nnvm.symbol.Variable("inputs")
        shape = (2, 16)
        target = tvm.target.create("llvm")
        z = nnvm.symbol.softmax(inputs)
        compute_graph = nnvm.graph.create(z)
        with nnvm.compiler.build_config(opt_level = 0):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs" : shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                m = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random((2, 16))
                m.set_input(**{"inputs":a_np})
                m.run()
                print(a_np)
                out = m.get_output(0, out = tvm.nd.empty((2, 16), ctx = ctx))
                print(out.asnumpy)
                print(compute_graph.ir())
                print(deploy_graph.ir())
def test_sigmoid():
        inputs = nnvm.symbol.Variable("inputs")
        shape = (2, 16)
        target = tvm.target.create("llvm")
        z = nnvm.symbol.sigmoid(inputs)
        compute_graph = nnvm.graph.create(z)
        with nnvm.compiler.build_config(opt_level = 0):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs" : shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                m = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random((2, 16))
                m.set_input(**{"inputs":a_np})
                m.run()
                print(a_np)
                out = m.get_output(0, out = tvm.nd.empty((2, 16), ctx = ctx))
                print(out.asnumpy)
                print(compute_graph.ir())
                print(deploy_graph.ir())

def test_relu():
        inputs = nnvm.symbol.Variable("inputs")
        shape = (2, 16)
        target = tvm.target.create("llvm")
        z = nnvm.symbol.relu(inputs)
        compute_graph = nnvm.graph.create(z)
        with nnvm.compiler.build_config(opt_level = 0):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs" : shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                m = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random((2, 16))
                m.set_input(**{"inputs":a_np})
                m.run()
                print(a_np)
                out = m.get_output(0, out = tvm.nd.empty((2, 16), ctx = ctx))
                print(out.asnumpy)
                print(compute_graph.ir())
                print(deploy_graph.ir())

def test_element_add():
        inputs1 = nnvm.symbol.Variable("inputs1")
        inputs2 = nnvm.symbol.Variable("inputs2")
        inputs3 = nnvm.symbol.Variable("inputs3")
        input_shape = (1, 24, 24, 16)
        out_shape = (1, 22, 22, 16)
        target = tvm.target.create("llvm")
        z1 = nnvm.symbol.conv2d(data=inputs1, channels = 16, kernel_size=(3, 3), padding = (0, 0), use_bias=False,
                                   layout='NHWC', kernel_layout='HWOI')
        # z2 = nnvm.symbol.elemwise_add(z1, inputs2)
        # z3 = nnvm.symbol.elemwise_mul(z2, inputs2)
        # z4 = nnvm.symbol.flatten(z3)
        # z5 = nnvm.symbol.elemwise_add(z4, inputs3)
        z2 = nnvm.symbol.relu(z1)
        z = nnvm.symbol.flatten(z2)
        # z = nnvm.symbol.global_max_pool2d(z1, layout='NHWC')
        
        
        compute_graph = nnvm.graph.create(z)
        with nnvm.compiler.build_config(opt_level = 1):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs1" : input_shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                m = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random((1, 24, 24, 16))
                b_np = np.random.random((1, 22, 22, 16))
                c_np = np.random.random((1, 7744))
                m.set_input(**{"inputs1":a_np})
                m.run()
                print("begin : ")
                out = m.get_output(0, out = tvm.nd.empty((1, 7744), ctx = ctx))
                print(out.asnumpy)
                print(compute_graph)
                print(deploy_graph.ir())

def test_element_sub():
        inputs1 = nnvm.symbol.Variable("inputs1")
        inputs2 = nnvm.symbol.Variable("inputs2")
        shape = (16, 16)
        target = tvm.target.create("llvm")
        z = nnvm.symbol.elemwise_sub(inputs1, inputs2)
        compute_graph = nnvm.graph.create(z)
        with nnvm.compiler.build_config(opt_level = 0):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs1" : shape, "inputs2" : shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                m = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random((16, 16))
                b_np = np.random.random((16, 16))
                m.set_input(**{"inputs1":a_np, "inputs2":b_np})
                m.run()
                print("begin : ")
                out = m.get_output(0, out = tvm.nd.empty((16, 16), ctx = ctx))
                print(out.asnumpy)
                print(compute_graph)
                print(deploy_graph)

def test_element_mul():
        inputs1 = nnvm.symbol.Variable("inputs1")
        inputs2 = nnvm.symbol.Variable("inputs2")
        shape = (16, 16)
        target = tvm.target.create("llvm")
        z = nnvm.symbol.elemwise_mul(inputs1, inputs2)
        compute_graph = nnvm.graph.create(z)
        with nnvm.compiler.build_config(opt_level = 0):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs1" : shape, "inputs2" : shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                m = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random((16, 16))
                b_np = np.random.random((16, 16))
                m.set_input(**{"inputs1":a_np, "inputs2":b_np})
                m.run()
                print("begin : ")
                out = m.get_output(0, out = tvm.nd.empty((16, 16), ctx = ctx))
                print(out.asnumpy)
                print(compute_graph)
                print(deploy_graph)

def test_pad():
        inputs = nnvm.symbol.Variable("inputs")
        shape = (2, 4)
        target = tvm.target.create("llvm")
        z = nnvm.symbol.pad(inputs, pad_width = ((2, 4),(4, 6)))
        compute_graph = nnvm.graph.create(z)         
        with nnvm.compiler.build_config(opt_level = 0):
                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = {"inputs":shape}, dtype = "float32")
                ctx = tvm.context(str(target), 0)
                m = runtime.create(deploy_graph, lib, ctx)
                a_np = np.random.random((2, 4))
                m.set_input(**{"inputs":a_np})
                m.run()
                out = m.get_output(0, out = tvm.nd.empty((8, 14), ctx = ctx))
                print(out)
                print(deploy_graph.ir())

def test_batch_norm():
    input_shape = (1, 224, 224, 3)
    target = tvm.target.create("llvm")
    inputs1 = nnvm.symbol.Variable("inputs1")
    z = nnvm.symbol.batch_norm(data = inputs1, axis = 3,epsilon = 2e-5, scale = False, name = "batch_norm")
    compute_graph = nnvm.graph.create(z)
        
    with nnvm.compiler.build_config(opt_level=0):
        deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs1" : input_shape}, dtype = "float32")
        ctx = tvm.context(str(target), 0)
        module = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.uniform(size  = (1, 224, 224, 3), low = -32, high = 32).astype(np.float32)
        print(a_np)
        module.set_input(inputs1 = a_np)
        module.run()
        out = module.get_output(0, out = tvm.nd.empty((1, 224, 224, 3)))
        print(out.asnumpy)
        print(compute_graph.ir())
        print(deploy_graph.ir())


def test_flatten():
    input_shape = (1, 2, 2, 2)
    target = tvm.target.create("llvm")
    inputs1 = nnvm.symbol.Variable("inputs1")
    z = nnvm.symbol.max_pool2d(inputs1, pool_size = [2, 2], layout = "NHWC")
    # z = nnvm.symbol.global_avg_pool2d(data = inputs1, layout = 'NHWC', name = "avg")
    compute_graph = nnvm.graph.create(z)
        
    with nnvm.compiler.build_config(opt_level=0):
        deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs1" : input_shape}, dtype = "float32")
        ctx = tvm.context(str(target), 0)
        module = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.uniform(size  = (1, 2, 2, 2), low = -32, high = 32).astype(np.float32)
        print(a_np)
        module.set_input(inputs1 = a_np)
        module.run()
        out = module.get_output(0, out = tvm.nd.empty((1, 1, 1, 2)))
        print(out.asnumpy)
        print(compute_graph.ir())
        print(deploy_graph.ir())
"""
print("test_conv2d      :        ")
test_conv2d()
print("test_dense       :        ")
test_dense()
print("test_softmax     :        ")
test_softmax()
print("test_relu        :        ")
test_relu() 
print("test_element_add :        ")
test_element_add()
print("test_pad         :        ")
test_pad()
test_batch_norm()
"""
test_dense()