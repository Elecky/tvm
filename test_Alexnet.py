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

def test_Alexnet():
    def Conv(data, kernel_size, filter_nums, stride = (1, 1), pad = (0, 0)):
        if pad[0] != 0 or pad[1] != 0:
            data = nnvm.symbol.pad(data = data, pad_width = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
        datas = nnvm.symbol.conv2d(data = data, kernel_size = kernel_size, channels = filter_nums, strides = stride,
                                        layout = 'NHWC', kernel_layout = 'HWOI')
        datas = nnvm.symbol.relu(data = datas)
        return datas
    def get_symbol(datas, num_classes):
        conv1 = Conv(data = datas, kernel_size = (11, 11), filter_nums = 96, stride = (4, 4))
        pool1 = nnvm.symbol.max_pool2d(data = conv1, pool_size = (3, 3), strides = (2, 2), layout = 'NHWC')
        conv2 = Conv(data = pool1, kernel_size = (5, 5), filter_nums = 256, pad = (2, 2))
        pool2 = nnvm.symbol.max_pool2d(data = conv2, pool_size = (3, 3), strides = (2, 2), layout = 'NHWC')
        conv3 = Conv(data = pool2, kernel_size = (3, 3), filter_nums = 384, pad = (1, 1))
        conv4 = Conv(data = conv3, kernel_size = (3, 3), filter_nums = 384, pad = (1, 1))
        conv5 = Conv(data = conv4, kernel_size = (3, 3), filter_nums = 256, pad = (1, 1))
        pool3 = nnvm.symbol.max_pool2d(data = conv5, pool_size = (3, 3), strides = (2, 2), layout = 'NHWC')
        datas = nnvm.symbol.flatten(data = pool3)
        fc1 = nnvm.symbol.dense(data = datas, units = 1024)
        relu1 = nnvm.symbol.relu(data = fc1)
        drop1 = nnvm.symbol.dropout(data = relu1, rate = 0.5)
        fc2 = nnvm.symbol.dense(data = drop1, units = 1024)
        relu2 = nnvm.symbol.relu(data = fc2)
        drop2 = nnvm.symbol.dropout(data = relu2, rate = 0.5)
        fc3 = nnvm.symbol.dense(data = drop2, units = 16)
        symbol = nnvm.symbol.softmax(fc3)
        return symbol

    input_shape = (1, 128, 128, 16)
    target_host = "llvm"
    device = "nnpu"
    data = nnvm.symbol.Variable(name = "data")
    target = tvm.target.create("llvm -device={}".format(device))
    print("ok")
    num_runs = 1
    z = get_symbol(datas = data, num_classes = 16)
    compute_graph = nnvm.graph.create(z)
    print(compute_graph.ir())
    with nnvm.compiler.build_config(opt_level = 0):
		if target.device_name != "nnpu":
			deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
											{"data" : input_shape}, dtype = "float32", target_host = target_host)
		else:
			with ScheduleProcHelper():
				with nnpu.build_config():
					nnpu.set_device(nnpu.get_env(), type = 'SC')
					deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
											{"data" : input_shape}, dtype = "float32", target_host = target_host)
		ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
		module = runtime.create(deploy_graph, lib, ctx)
		a_np = np.random.randint(size  = input_shape, low = -32, high = 32)
		print(a_np)
		module.set_input(data = a_np)
		ftimer = module.module.time_evaluator("run", ctx, number = num_runs, repeat = 1)
		# module.run()
		out = module.get_output(0, out = tvm.nd.empty((1, 16)))
		print(out.asnumpy)
		print(deploy_graph.ir())
		print(ftimer().mean * 10)
    
test_Alexnet()


