from __future__ import absolute_import, print_function
import nnvm.symbol as sym
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
def test_vgg():
	def get_feature(internel_layer, layers, filters, batch_norm = False):
		"""
		Get VGG feature body as stacks of convoltions.
		layers  : [1, 1, 2, 2, 2]
		filters : [64, 128, 256, 512, 512]
		"""
		for i, num in enumerate(layers):
			"""
			i = 0, num = 1
			i = 1, num = 1
			i = 2, num = 2
			i = 3, num = 2
			i = 4, num = 2
			"""
			for j in range(num):
				internel_layer = sym.pad(data = internel_layer, pad_width = ((0, 0), (1, 1), (1, 1), (0, 0)))
				internel_layer = sym.conv2d(data = internel_layer, kernel_size = (3, 3), 
												channels = filters[i], layout = 'NHWC', kernel_layout = 'HWOI',
													name = "conv%s_%s"%(i+1, j+1))
				if batch_norm:
					internel_layer = sym.batch_norm(data = internel_layer, axis = 3, name = "bn%s_%s"%(i+1, j+1))
				internel_layer = sym.relu(data = internel_layer, name = "relu%s_%s"%(i+1, j+1))
				
			internel_layer = sym.max_pool2d(data = internel_layer, pool_size = (2, 2), strides = (2, 2), layout = "NHWC",
												name = "pool%s"%(i+1))
			return internel_layer

	def get_classifier(input_data, num_classes):
		"""
		Get VGG classifier layers as fc layers.
		"""
		flatten = sym.flatten(data = input_data, name = "flatten")
		fc1 = sym.dense(data = flatten, units = 32, name = "fc1")
		relu1 = sym.relu(data = fc1, name = "relu1")
		drop1 = sym.dropout(data = relu1, rate = 0.5, name = "drop1")
		fc2 = sym.dense(data = drop1, units = 32, name = "fc2")
		relu2 = sym.relu(data = fc2, name = "relu2")
		drop2 = sym.dropout(data = relu2, rate = 0.5, name = "drop2")
		fc3 = sym.dense(data = drop2, units = num_classes, name = "fc3")
		return fc3
	def get_symbol(datas, num_classes, num_layers = 11, batch_norm = False):
		"""
		Parameters
		------------
		num_classes     : int, default 16
						Number of classification classes

		num_layers      : int
						Number of layers for the variant of vgg. Options are 11, 13, 16, 19

		batch_norm      : bool, default False
						Use batch normalization.

		"""
		vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
					13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
					16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
					19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}

		if num_layers not in vgg_spec:
			raise ValueError("Invalide num_layers {}. Choices are 11, 13, 16, 19.".format(num_layers))
		layers, filters = vgg_spec[num_layers]
		feature = get_feature(datas, layers, filters, batch_norm)
		classifier = get_classifier(feature, num_classes)
		symbol = sym.softmax(data = classifier, name = "softmax")
		return symbol
	
	input_shape = (1, 224, 224, 16)
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
			nnpu.set_device(nnpu.get_env(), type = 'SC')
			with ScheduleProcHelper():
				with nnpu.build_config():
					deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
											{"data" : input_shape}, dtype = "float32", target_host = target_host)
		ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
		module = runtime.create(deploy_graph, lib, ctx)
		a_np = np.random.uniform(size  = input_shape, low = -32, high = 32).astype(np.float32)
		print(a_np)
		module.set_input(data = a_np)
		ftimer = module.module.time_evaluator("run", ctx, number = num_runs, repeat = 1)
		# module.run()
		out = module.get_output(0, out = tvm.nd.empty((1, 16)))
		print(out.asnumpy)
		print(deploy_graph.ir())
		print(ftimer().mean * 10)
        
test_vgg()