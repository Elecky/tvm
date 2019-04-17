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

def test_inception_v3():
    def Conv(data, num_filter, kernel = (1, 1), stride = (1, 1), pad = (0, 0), name = None, suffix = ''):
        if pad[0] != 0 or pad[1] != 0:
            data = sym.pad(data = data, pad_width = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
        conv = sym.conv2d(data = data, channels = num_filter, kernel_size = kernel, strides = stride,
                            padding = (0, 0), use_bias = False, layout = 'NHWC', kernel_layout = 'HWOI',
                                name = "%s%s_conv2d"%(name, suffix))
        bn = sym.batch_norm(data = conv, name = "%s%s_batchnorm"%(name, suffix), epsilon = 2e-5, axis = 3)
        act = sym.relu(data = bn, name = "%s%s_relu"%(name, suffix))
        return act
    def Pooling(data, kernel, stride, pad, pool_type, name):
        if pad[0] != 0 or pad[1] != 0:
            data = sym.pad(data = data, pad_width = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
        if pool_type == 'max':
            return sym.max_pool2d(data = data, pool_size = kernel, strides = stride, name = name, layout = 'NHWC')
        if pool_type == 'avg':
            return sym.avg_pool2d(data = data, pool_size = kernel, strides = stride, name = name, layout = 'NHWC')
        raise ValueError("Invalid pooling type: " + pool_type)

    def Inception7A(data, 
                    num_1x1,
                    num_3x3_red, num_3x3_1, num_3x3_2,
                    num_5x5_red, num_5x5,
                    pool, proj, name):
        # num_1x1 = 64
        # num_3x3_red = 64
        # num_3x3_1 = 96
        # num_3x3_2 = 96
        # num_5x5_red = 48
        # num_5x5 = 64
        tower_1x1 = Conv(data, num_1x1, name = ('%s_conv'%name))

        tower_5x5 = Conv(data, num_5x5_red, name = ('%s_tower'%name), suffix = '_conv')
        tower_5x5 = Conv(tower_5x5, num_5x5, kernel = (5, 5), pad = (2, 2), name = ('%s_tower'%name), 
                            suffix = '_conv_1')

        tower_3x3 = Conv(data, num_3x3_red, name = ('%s_tower_1'%name), suffix = "_conv")
        tower_3x3 = Conv(tower_3x3, num_3x3_1, kernel = (3, 3), pad = (1, 1), name = ('%s_tower_1'%name),
                            suffix = '_conv_1')
        tower_3x3 = Conv(tower_3x3, num_3x3_2, kernel = (3, 3), pad = (1, 1), name = ('%s_tower_1'%name),
                            suffix = '_conv_2')
        pooling = Pooling(data, kernel = (3, 3), stride = (1, 1), pad = (1, 1), pool_type = pool,
                            name =('%s_pool_%s_pool'%(pool, name)))

        cproj = Conv(pooling, proj, name = ('%s_tower_2'%name), suffix = '_conv')
        concat = sym.concatenate(*[tower_1x1, tower_5x5, tower_3x3, cproj], axis = 3, name = 'ch_concat_%s_chconcat'%name)
        return concat

    def Inception7B(data, 
                    num_3x3,
                    num_d3x3_red, num_d3x3_1, num_d3x3_2,
                    pool,
                    name):
        tower_3x3 = Conv(data, num_3x3, kernel = (3, 3), pad = (0, 0), stride = (2, 2), 
                            name = ('%s_conv'%name))

        tower_d3x3 = Conv(data, num_d3x3_red, name = ('%s_tower'%name), suffix = '_conv')
        tower_d3x3 = Conv(tower_d3x3, num_d3x3_1, kernel = (3, 3), pad = (1, 1), stride = (1, 1),
                        name=('%s_tower'%name), suffix = '_conv_1')
        tower_d3x3 = Conv(tower_d3x3, num_d3x3_2, kernel = (3, 3), pad = (0, 0), stride = (2, 2),
                        name = ('%s_tower'%name), suffix = '_conv_2')

        pooling = Pooling(data = data, kernel = (3, 3), stride = (2, 2), pad = (0, 0), pool_type = "max",
                        name = ('max_pool_%s_pool'%name))
        concat = sym.concatenate(*[tower_3x3, tower_d3x3, pooling], axis = 3, name='ch_concat_%s_chconcat'%name)
        return concat

    def Inception7C(data,
                    num_1x1,
                    num_d7_red, num_d7_1, num_d7_2,
                    num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4,
                    pool, proj,
                    name):
        tower_1x1 = Conv(data = data, num_filter = num_1x1, kernel = (1, 1), name = ('%s_conv'%name))


        tower_d7 = Conv(data = data, num_filter = num_d7_red, name = ('%s_tower'%name), suffix = '_conv')
        tower_d7 = Conv(data = tower_d7, num_filter = num_d7_1, kernel = (1, 7), pad = (0, 3),
                        name = ('%s_tower'%name), suffix = '_conv_1')
        tower_d7 = Conv(data = tower_d7, num_filter = num_d7_2, kernel = (7, 1), pad = (3, 0),
                        name = ('%s_tower' % name), suffix = '_conv_2')


        tower_q7 = Conv(data = data, num_filter = num_q7_red, name = ('%s_tower_1'%name), suffix = '_conv')
        tower_q7 = Conv(data = tower_q7, num_filter = num_q7_1, kernel = (7, 1), pad = (3, 0),
                        name = ('%s_tower_1'%name), suffix = '_conv_1')
        tower_q7 = Conv(data = tower_q7, num_filter = num_q7_2, kernel = (1, 7), pad = (0, 3),
                        name = ('%s_tower_1'%name), suffix = '_conv_2')
        tower_q7 = Conv(data = tower_q7, num_filter = num_q7_3, kernel = (7, 1), pad = (3, 0),
                        name = ('%s_tower_1'%name), suffix = '_conv_3')
        tower_q7 = Conv(data = tower_q7, num_filter = num_q7_4, kernel = (1, 7), pad = (0, 3),
                        name = ('%s_tower_1'%name), suffix = '_conv_4')


        pooling = Pooling(data = data, kernel = (3, 3), stride = (1, 1), pad = (1, 1), pool_type = pool,
                        name = ('%s_pool_%s_pool'%(pool, name)))
        cproj = Conv(data = pooling, num_filter = proj, kernel = (1, 1),
                    name=('%s_tower_2'%name), suffix = '_conv')

        concat = sym.concatenate(*[tower_1x1, tower_d7, tower_q7, cproj], axis = 3, 
                                name='ch_concat_%s_chconcat'%name)
        return concat


    def Inception7D(data,
                    num_3x3_red, num_3x3,
                    num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3,
                    pool,
                    name):
        tower_3x3 = Conv(data = data, num_filter = num_3x3_red, name = ('%s_tower'%name),
                        suffix = '_conv')
        tower_3x3 = Conv(data = tower_3x3, num_filter = num_3x3, kernel = (3, 3), pad = (0, 0), stride = (2, 2),
                        name = ('%s_tower'%name), suffix = '_conv_1')
        

        tower_d7_3x3 = Conv(data = data, num_filter = num_d7_3x3_red, name = ('%s_tower_1'%name),
                            suffix = '_conv')
        tower_d7_3x3 = Conv(data = tower_d7_3x3, num_filter = num_d7_1, kernel = (1, 7), pad = (0, 3),
                            name = ('%s_tower_1'%name), suffix = '_conv_1')
        tower_d7_3x3 = Conv(data = tower_d7_3x3, num_filter = num_d7_2, kernel = (7, 1), pad = (3, 0),
                            name = ('%s_tower_1'%name), suffix = '_conv_2')
        tower_d7_3x3 = Conv(data = tower_d7_3x3, num_filter = num_d7_3x3, kernel = (3, 3), stride = (2, 2),
                            name = ('%s_tower_1'%name), suffix = '_conv_3')
        pooling = Pooling(data = data, kernel = (3, 3), stride = (2, 2), pool_type = pool, pad = (0, 0),
                        name = ('%s_pool_%s_pool'%(pool, name)))


        concat = sym.concatenate(*[tower_3x3, tower_d7_3x3, pooling], axis = 3,
                                name='ch_concat_%s_chconcat' % name)
        return concat

    def Inception7E(data,
                    num_1x1,
                    num_d3_red, num_d3_1, num_d3_2,
                    num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2,
                    pool, proj,
                    name):

        tower_1x1 = Conv(data = data, num_filter = num_1x1, kernel = (1, 1), name = ('%s_conv'%name))


        tower_d3 = Conv(data = data, num_filter = num_d3_red, name = ('%s_tower'%name), suffix = '_conv')
        tower_d3_a = Conv(data = tower_d3, num_filter = num_d3_1, kernel = (1, 3), pad=(0, 1),
                        name = ('%s_tower'%name), suffix = '_mixed_conv')
        tower_d3_b = Conv(data = tower_d3, num_filter = num_d3_2, kernel = (3, 1), pad = (1, 0),
                        name = ('%s_tower'%name), suffix = '_mixed_conv_1')


        tower_3x3_d3 = Conv(data = data, num_filter = num_3x3_d3_red, name = ('%s_tower_1'%name),
                            suffix = '_conv')
        tower_3x3_d3 = Conv(data = tower_3x3_d3, num_filter = num_3x3, kernel = (3, 3), pad = (1, 1),
                            name = ('%s_tower_1'%name), suffix = '_conv_1')
        tower_3x3_d3_a = Conv(data = tower_3x3_d3, num_filter = num_3x3_d3_1, kernel = (1, 3), pad = (0, 1),
                            name = ('%s_tower_1'%name), suffix = '_mixed_conv')
        tower_3x3_d3_b = Conv(data = tower_3x3_d3, num_filter = num_3x3_d3_2, kernel = (3, 1), pad = (1, 0),
                            name = ('%s_tower_1'%name), suffix = '_mixed_conv_1')


        pooling = Pooling(data = data, kernel = (3, 3), stride = (1, 1), pad = (1, 1), pool_type = pool,
                        name = ('%s_pool_%s_pool'%(pool, name)))
        cproj = Conv(data = pooling, num_filter = proj, kernel = (1, 1), name = ('%s_tower_2'%name),
                    suffix = '_conv')

        concat = sym.concatenate(
            *[tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj], axis = 3, 
            name = 'ch_concat_%s_chconcat'%name)
        return concat
    def get_symbol(data, num_classes = 16, **kwargs):
        conv = Conv(data, 32, kernel = (3, 3), stride = (2, 2), name = "conv")
        conv_1 = Conv(conv, 32, kernel = (3, 3), name = "conv_1")
        conv_2 = Conv(conv_1, 64, kernel = (3, 3), pad = (1, 1), name = "conv_2")
        pool = Pooling(data = conv_2, kernel = (3, 3), stride = (2, 2), pool_type = "max", pad = (0, 0), 
                        name = "pool")
        conv_3 = Conv(pool, 80, kernel = (1, 1), name = "conv_3")
        conv_4 = Conv(conv_3, 192, kernel = (3, 3), name = "conv_4")
        pool1 = Pooling(data = conv_4, kernel = (3, 3), stride = (2, 2), pool_type = "max", pad = (0, 0), 
                        name = "pool1")

        in3a = Inception7A(pool1, 64, 
                            64, 96, 96, 
                            48, 64, 
                            "avg", 32, "mixed")
        in3b = Inception7A(in3a, 64, 
                            64, 96, 96, 
                            48, 64, 
                            "avg", 64, "mixed_1")

        in3c = Inception7A(in3b, 64, 
                            64, 96, 96, 
                            48, 64, 
                            "avg", 64, "mixed_2")

        in3d = Inception7B(in3c, 384, 
                            64, 96, 96, 
                            "max", "mixed_3")

        in4a = Inception7C(in3d, 192,
                            128, 128, 192,
                            128, 128, 128, 128, 192,
                            "avg", 192, "mixed_4")

        in4b = Inception7C(in4a, 192,
                            160, 160, 192, 
                            160, 160, 160, 160, 192,
                            "avg", 192, "mixed_5")
        in4c = Inception7C(in4b, 192, 
                            160, 160, 192,
                            160, 160, 160, 160, 192,
                            "avg", 192, "mixed_6")
        in4d = Inception7C(in4c, 192, 
                            192, 192, 192,
                            192, 192, 192, 192, 192,
                            "avg", 192, "mixed_7")

        in4e = Inception7D(in4d, 192, 320,
                            192, 192, 192, 192,
                            "max", "mixed_8")

        in5a = Inception7E(in4e, 320,
                            384, 384, 384,
                            448, 384, 384, 384,
                            "avg", 192, "mixed_9")
        in5b = Inception7E(in5a, 320,
                            384, 384, 384,
                            448, 384, 384, 384,
                            "max", 192, "mixed_10")

        pool = Pooling(data = in5b, kernel = (8, 8), stride = (1, 1), pool_type = "avg", pad = (0, 0),
                        name = "global_pool")
        flatten = sym.flatten(data = pool, name = "flatten")
        fc1 = sym.dense(data = flatten, units = num_classes, name = "fc1")
        softmax = sym.softmax(data = fc1, name = "softmax")
        return softmax
    
    input_shape = (1, 299, 299, 16)
    target_host = "llvm"
    device = "nnpu"
    data = nnvm.symbol.Variable(name = "data")
    target = tvm.target.create("llvm -device={}".format(device))
    print("ok")
    num_runs = 3
    z = get_symbol(data = data, num_classes = 16)
    compute_graph = nnvm.graph.create(z)
    with nnvm.compiler.build_config(opt_level = 1):
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
		a_np = np.random.uniform(size  = (1, 299, 299, 16), low = -32, high = 32).astype(np.float32)
		print(a_np)
		module.set_input(data = a_np)
		ftimer = module.module.time_evaluator("run", ctx, number = num_runs, repeat = 1)
		module.run()
		out = module.get_output(0, out = tvm.nd.empty((1, 16)))
		print(out.asnumpy)
		print(deploy_graph.ir())
		print(ftimer().mean * 10)
test_inception_v3()