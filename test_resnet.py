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

def test_resnets():
    def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck = True):
        # bottle_neck : False
        """
        Return Resnet Unit symbol for building Resnet
        Parameters
        ------------
        data       :  str
                    Input data
        
        num_filter :  int
                    Number of output channels
        
        stride     :  tuple
                    Stride used in convolution
                
        dim_match  :  Boolean
                    True means channel number between input and output is the same
                    otherwise means differ
        """
        if bottle_neck:
            bn1 = nnvm.symbol.batch_norm(data = data, axis = 3, epsilon = 2e-5, name = name + '_bn1')
            act1 = nnvm.symbol.relu(data = bn1, name = name + '_relu1')
            conv1 = nnvm.symbol.conv2d(data = act1, channels = int(num_filter * 0.25), kernel_size = (1, 1),
                                strides = stride, padding = (0, 0), use_bias = False, layout = 'NHWC',
                                    kernel_layout = 'HWOI', name = name + '_conv1')
            bn2 = nnvm.symbol.batch_norm(data = conv1, axis = 3, epsilon = 2e-5, name = name + '_bn2')
            act2 = nnvm.symbol.relu(data = bn2, name = name + '_relu2')
            
            pad = nnvm.symbol.pad(data = act2, pad_width = ((0, 0), (1, 1), (1, 1), (0, 0)))
            conv2 = nnvm.symbol.conv2d(data = pad, channels = int(num_filter * 0.25), kernel_size = (3, 3), 
                                strides = (1, 1), padding = (0, 0), use_bias = False, layout = 'NHWC',
                                    kernel_layout = 'HWOI', name = name + '_conv2')
                                    
            bn3 = nnvm.symbol.batch_norm(data = conv2, axis = 3, epsilon = 2e-5, name = name+ '_bn3')
            act3 = nnvm.symbol.relu(data = bn3, name = name +'_relu3')
            conv3 = nnvm.symbol.conv2d(data = act3, channels = num_filter, kernel_size = (1, 1), strides = (1, 1),
                                padding = (0, 0), use_bias = False, layout = 'NHWC', kernel_layout = 'HWOI',
                                    name = name + '_conv3')
                                    
            if dim_match:
                shortcut = data
            else:
                shortcut = nnvm.symbol.conv2d(data = act1, channels = num_filter, kernel_size = (1, 1), strides = stride,
                                        use_bias = False, layout = 'NHWC', kernel_layout = 'HWOI', name = name + '_sc')
            return nnvm.symbol.elemwise_add(conv3, shortcut)
        else:
            # bottle_neck = False
            # i = 0 : filter_list[1] = 64, (1, 1), False
            # i = 1 : filter_list[2] = 128, (2, 2), False
            # i = 2 : filter_list[3] = 256, (2, 2), False
            # i = 3 : filter_list[4] = 512, (2, 2), False
            # bn1 = nnvm.symbol.batch_norm(data = data, axis = 3, epsilon = 2e-5, name = name + '_bn1')
            act1 = nnvm.symbol.relu(data = data, name = name + '_relu1')
            # (56, 56, 64)
            # num_filter = filter_list[1] = 64
            # strides = (1, 1)
            pad1 = nnvm.symbol.pad(data = act1, pad_width = ((0, 0), (1, 1), (1, 1), (0, 0)))
            conv1 = nnvm.symbol.conv2d(data = pad1, channels = num_filter, kernel_size = (3, 3), strides = stride,
                                padding = (0, 0), use_bias = False, layout = 'NHWC', kernel_layout = 'HWOI',
                                    name = name +'_bn2')
            # i = 0 : (56, 56, 64)
            # i = 1 : (28, 28, 128)
            # i = 2 : (14, 14, 256)
            # i = 3 : (7, 7, 512)
            # bn2 = nnvm.symbol.batch_norm(data = conv1, axis = 3, epsilon = 2e-5, name = name + '_bn2')
            act2 = nnvm.symbol.relu(data = conv1, name = name + '_relu2')
            # (56, 56, 64)
            pad2 = nnvm.symbol.pad(data = act2, pad_width = ((0, 0), (1, 1), (1, 1), (0, 0)))
            conv2 = nnvm.symbol.conv2d(data = pad2, channels = num_filter, kernel_size = (3, 3), strides = (1, 1), 
                                padding = (0, 0), use_bias = False, layout = 'NHWC', kernel_layout = 'HWOI', 
                                    name = name + '_conv2')
            # i = 0 : (56, 56, 64)
            # i = 1 : (28, 28, 128)
            # i = 2 : (14, 14, 256)
            if dim_match:
                shortcut = data
            else:
                shortcut = nnvm.symbol.conv2d(data = act1, channels = num_filter, kernel_size = (1, 1), strides = stride,
                                        use_bias = False, layout = 'NHWC', kernel_layout = 'HWOI',
                                            name = name + '_sc')
            return nnvm.symbol.elemwise_add(conv2, shortcut)



    def resnet(datas, units, num_stages, filter_list, num_classes, image_shape, bottle_neck = True):
        # units = [2, 2, 2, 2]
        # num_stages = 4
        # filter_list = [64, 64, 128, 256, 512]
        # num_classes = 1000
        # image_shape = (224, 224, 16)
        # bottle_neck = False
        """
        Return Resnet symbol of
        Parameters
        ------------
        units       : list
                        Number of units in each stage
        
        num_stage   : int
                        Number of stage
                        
        filter_list : list
                        Channel size of each stage
        
        num_classes : int
                        Output size of symbol
        """
        num_unit = len(units)
        assert num_unit == num_stages
        
        data = nnvm.symbol.batch_norm(data = datas, axis = 3, epsilon = 2e-5, scale = False, name = "bn_data")

        (_, height, _, _) = image_shape
        if height <= 32:
            pad = nnvm.symbol.pad(data = data, pad_width = ((0, 0), (1, 1), (1, 1), (0, 0)))
            body = nnvm.symbol.conv2d(data = pad, channels = filter_list[0], kernel_size = (3,3), 
                                strides = (1, 1), padding = (1, 1), use_bias = False, layout = 'NHWC', 
                                    kernel_layout = 'HWOI',name = "conv0")
        else:
            pad = nnvm.symbol.pad(data = data, pad_width = ((0, 0), (3, 3), (3, 3),(0, 0)))
    
            body = nnvm.symbol.conv2d(data = pad, channels = filter_list[0], kernel_size = (7, 7),
                                strides = (2, 2), padding = (0, 0), use_bias = False, layout = 'NHWC',
                                    kernel_layout = 'HWOI', name = "conv0")
            # body.shape = (112, 112, 64)
            # body = nnvm.symbol.batch_norm(data = body, axis = 3, epsilon = 2e-5, name = "bn0")
            body = nnvm.symbol.relu(data = body, name = "relu0")
            # body = nnvm.symbol.pad(data = body, pad_width = ((0, 0), (1, 1), (1, 1), (0, 0)))
            body = nnvm.symbol.max_pool2d(data = body, pool_size = (3, 3), strides = (2, 2), layout = 'NHWC')

            # body.shape = (56, 56, 64)
        for i in range(num_stages):
            # num_stages == 4
            # i = 0: (56, 56, 64)
            # i = 1: filter_list[2] = 128, (2, 2), False (28, 28, 128)
            # i = 2: filter_list[3] = 256, (2, 2), False
            # i = 3: filter_list[4] = 512, (2, 2), False
            
            body = residual_unit(body, filter_list[i+1], (1 if i == 0 else 2, 1 if i == 0 else 2), False, 
                                name = 'stage%d_unit%d'%(i+1, 1), bottle_neck = bottle_neck)
            # (56, 56, 64)
            # units[0] - 1 = 1
            for j in range(units[i] - 1):
                body = residual_unit(body, filter_list[i+1], (1, 1), True, name = "stage%d_unit%d"%(i+1, j+2), 
                                    bottle_neck = bottle_neck)
                # (56, 56, 64) 
        # (7, 7, 512)
        # bn1 = nnvm.symbol.batch_norm(data = body, axis = 3, epsilon = 2e-5, name = "bn1")
        relu1 = nnvm.symbol.relu(data = body, name = "relu1")
        pool1 = nnvm.symbol.global_avg_pool2d(data = relu1, layout = 'NHWC', name = "pool1")
        # (1, 1, 512)
        flat = nnvm.symbol.flatten(data = pool1)
        # (512)
        fc1 = nnvm.symbol.dense(data = flat, units = num_classes, name = 'fc1')
        
        return nnvm.symbol.softmax(data = fc1, name = 'softmax')
    def get_symbol(datas, num_classes, num_layers = 50, image_shape = (1, 224, 224, 16), **kwargs):
        (_, height, _, _) = image_shape
        if height <= 28:
            num_stages = 3
            if (num_layers - 2) % 9 == 0 and num_layers >= 164:
                per_unit = [(num_layers - 2) // 9]
                filter_list = [16, 64, 128, 256]
                bottle_neck = True
            elif (num_layers - 2) % 6 == 0 and num_layers < 164:
                per_unit = [(num_layers - 2) // 6]
                filter_list = [16, 16, 32, 64]
                bottle_neck = False
            else:
                raise ValueError("no experiments done on num_layers {}".format(num_layers))
            units = per_unit * num_stages
        else:
            print("height = 224 > 28")
            # height = 224 > 28
            if num_layers >= 50:
                filter_list = [64, 256, 512, 1024, 2048]
                
                bottle_neck = True
            else:
                print("num_layers = 18 < 50")
                # num_layers = 18 < 50
                filter_list = [64, 64, 128, 256, 512]
                bottle_neck = False
            num_stages = 4
            if num_layers == 18:
                units = [2, 2, 2, 2]
            elif num_layers == 34:
                units = [3, 4, 6, 3]
            elif num_layers == 50:
                units = [3, 4, 6, 3]
            elif num_layers == 101:
                units = [3, 4, 23, 3]
            elif num_layers == 152:
                units = [3, 8, 36, 3]
            elif num_layers == 200:
                units = [3, 24, 36, 3]
            elif num_layers == 269:
                units = [3, 30, 48, 8]
            else:
                raise ValueError("no experiments done on num_layers {}".format(num_layers))
            return resnet(datas = datas, units = units, num_stages = num_stages, filter_list = filter_list, num_classes = num_classes, image_shape = image_shape, bottle_neck = bottle_neck)

    input_shape = (1, 224, 224, 16)
    target_host = "llvm"
    device = "nnpu"
    data = nnvm.symbol.Variable(name = "data")
    target = tvm.target.create("llvm -device={}".format(device))
    print("ok")
    num_runs = 3
    z = get_symbol(datas = data, num_classes = 16, num_layers = 18, image_shape = (1, 224, 224, 16))
    compute_graph = nnvm.graph.create(z)
    print(compute_graph.ir())
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
        print(deploy_graph.ir())
        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        module = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.uniform(size  = (1, 224, 224, 16), low = -32, high = 32).astype(np.float32)
        print(a_np)
        module.set_input(data = a_np)
        ftimer = module.module.time_evaluator("run", ctx, number = num_runs, repeat = 1)
        
        module.run()
        out = module.get_output(0, out = tvm.nd.empty((1, 16)))
        print(out.asnumpy)
        print(deploy_graph.ir())
        print(ftimer().mean * 10)
        
test_resnets()