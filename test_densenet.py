from __future__ import absolute_import, print_function
import nnvm
import numpy as np 

import nnvm.compiler
from tvm.contrib import graph_runtime as runtime
from nnvm.testing import utils
import nnvm.testing
import nnpu
from nnpu.utils import ScheduleProcHelper
import tvm
import logging
from collections import namedtuple
import time
logging.basicConfig()
def test_densenet():
    def Conv(datas, kernel_size, filter_nums, stride = (1, 1), pad = (0, 0)):
        if pad[0] != 0 or pad[1] != 0:
            datas = nnvm.symbol.pad(data = datas, pad_width = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
        conv = nnvm.symbol.conv2d(data = datas, kernel_size = kernel_size, channels = filter_nums, strides = stride, 
                layout = 'NHWC', kernel_layout = 'HWOI')
        return conv
    def bottleneck_layer(datas, filters):
        bn1 = nnvm.symbol.batch_norm(data = datas, epsilon = 2e-5, axis = 3)
        relu1 = nnvm.symbol.relu(data = bn1)
        conv1 = Conv(datas = relu1, kernel_size = (1, 1), filter_nums = 4 * filters)
        bn2 = nnvm.symbol.batch_norm(data = conv1, epsilon = 2e-5, axis = 3)
        relu2 = nnvm.symbol.relu(data = bn2)
        conv2 = Conv(datas = relu2, kernel_size = (3, 3), filter_nums = filters, pad = (1, 1))
        return conv2
    def transition_layer(datas, filters):
        conv = Conv(datas = datas, kernel_size = (1, 1), filter_nums = filters)
        
        pool = nnvm.symbol.avg_pool2d(data = conv, pool_size = (2, 2), strides = (2, 2), layout = 'NHWC')
        return pool

    def dense_block(datas, filters, layers):
        layers_concat = []
        layers_concat.append(datas)
        b_l = bottleneck_layer(datas, filters)
        
        layers_concat.append(b_l)
        for i in range(layers - 1):
            x = nnvm.symbol.concatenate(*layers_concat, axis = 3)
            x = bottleneck_layer(x, filters)
            layers_concat.append(x)
        return x


    def get_symbol(datas, num_classes = 16):
        x = Conv(datas, kernel_size = (7, 7), filter_nums = 32, stride = (2, 2))
        
        x = nnvm.symbol.max_pool2d(x, pool_size = (3, 3), strides = (2, 2), layout = 'NHWC')
        
        b1 = dense_block(x, 32, 6)
        
        l1 = transition_layer(b1, 32)
        
        b2 = dense_block(l1, 32, 12)
        l2 = transition_layer(b2, 32)
        b3 = dense_block(l2, 32, 48)
        l3 = transition_layer(b3, 32)
        b4 = dense_block(l3, 32, 32)
        x = nnvm.symbol.global_avg_pool2d(data = b4, layout = 'NHWC')
        x = nnvm.symbol.flatten(data = x)
        fc = nnvm.symbol.dense(data = x, units = 16)
        symbol = nnvm.symbol.softmax(data = fc)
        return symbol
    input_shape = (1, 229, 229, 16)
    target_host = "llvm"
    device = "nnpu"
    data = nnvm.symbol.Variable(name = "data")
    target = tvm.target.create("llvm -device={}".format(device))
    print("ok")
    num_runs = 3
    z = get_symbol(datas = data, num_classes = 16)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 0):
		if target.device_name != "nnpu":
			deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
											{"data" : input_shape}, dtype = "float32", target_host = target_host)
		else:
			with ScheduleProcHelper():
				with nnpu.build_config():
					nnpu.set_device(nnpu.get_env(), type = 'S0')
					deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
											{"data" : input_shape}, dtype = "float32", target_host = target_host)
		ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
		module = runtime.create(deploy_graph, lib, ctx)
		a_np = np.random.random(size  = input_shape)
		print(a_np)
		module.set_input(data = a_np)
		ftimer = module.module.time_evaluator("run", ctx, number = num_runs, repeat = 1)
		module.run()

		out = module.get_output(0, out = tvm.nd.empty((1, 16)))
		print(out.asnumpy)
		print(deploy_graph.ir())
		print(ftimer().mean)
test_densenet()