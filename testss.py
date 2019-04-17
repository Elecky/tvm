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
import nnpu
from nnpu.utils import ScheduleProcHelper
logging.basicConfig()


def test_batch_norm():
    input_shape = (1, 4, 4, 16)
    target_host = "llvm"
    device = "nnpu"
    target = tvm.target.create("llvm -device={}".format(device))
    inputs1 = nnvm.symbol.Variable("inputs1")
    inputs2 = nnvm.symbol.Variable("inputs2")
    z1 = nnvm.symbol.relu(inputs1)
    # z2 = nnvm.symbol.relu(z1)
    compute_graph = nnvm.graph.create(z1)
        
    with nnvm.compiler.build_config(opt_level=0):
        if target.device_name != "nnpu":

                deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs1" : input_shape}, dtype = "float32", target_host = target_host)
        else:
            with ScheduleProcHelper():
                with nnpu.build_config():
                    nnpu.set_device(nnpu.get_env(), type = 'S0')
                    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target, shape = 
                                        {"inputs1" : input_shape}, dtype = "float32", target_host = target_host)

        ctx = tvm.context(str("nnpu"), 0) if device == "nnpu" else tvm.context(str("llvm"), 0)
        module = runtime.create(deploy_graph, lib, ctx)
        a_np = np.random.uniform(size  = (1, 4, 4, 16), low = -32, high = 32).astype(np.float32)
        b_np = np.random.uniform(size  = (1, 16), low = -32, high = 32).astype(np.float32)
        print(a_np)
        module.set_input(inputs1 = a_np)
        module.run()
        out = module.get_output(0, out = tvm.nd.empty((1, 4, 4, 16)))
        print(out.asnumpy)
        print(compute_graph.ir())
        print(deploy_graph.ir())
print("begin : ")
test_batch_norm()
print("end   ")

from __future__ import absolute_import
import tvm
import nnpu
from nnpu.utils import ScheduleProcHelper
import topi
import numpy as np

def matmul(lhs, rhs, transpose_a = 0, transpose_b = 0):
	"""
	matmul
	Parameters
	------------
	lhs : tvm.tensor
	      n-D dimension
	rhs : tvm.tensor
	      n-D 
	transpose_a : optional, boolean, default = 0
	transpose_b : optional, boolean, default = 0


	1-D arrays: inner product of vectors

	2-D arrays: matrix multiplication


	Returns 
	------------
	output : tvm.tensor
	         output shape is the same as input(lhs and rhs shape)
	         
	"""

	factor = 16
	env = nnpu.get_env()
	assert len(lhs.shape) == len(rhs.shape) 
	assert len(lhs.shape) == 1 or len(lhs.shape) == 2
	gemm_shape = (1, 16, 16)
	if lhs.dtype == rhs.dtype == env.cfg['dtype_n']:
		modes = 'n'
	elif lhs.dtype == rhs.dtype == env.cfg['dtype_w']:
		modes = 'w'
	if transpose_a != 0:
		s = [i for i in range(len(lhs.shape))]
		s.reverse()
		axis = tuple(s)
		lhs = nnpu.utils.transpose(lhs, axis) 
	if transpose_b != 0:
		s = [i for i in range(len(rhs.shape))]
		s.reverse()
		axis = tuple(s)
		rhs = nnpu.utils.transpose(rhs, axis) 
	if len(lhs.shape) == 1:
		k = tvm.reduce_axis((0, lhs.shape[0]))
		res = tvm.compute((1, ), tvm.sum(lhs(k).astype(env.cfg['dtype_w']) * rhs(k).astype(env.cfg['dtype_w']), axis = k))
	else:
		k = tvm.reduce_axis((0, lhs.shape[1]))
		s = [i for i in range(len(rhs.shape))]
		s.reverse()
		axis = tuple(s)
		rhs = nnpu.utils.transpose(rhs, axis) 
		print(lhs.shape)
		print(rhs.shape)
		res_buf = tvm.compute((lhs.shape[0], rhs.shape[1]), lambda i, j : tvm.sum(lhs[i, k].astype(env.cfg['dtype_w']) * rhs[j, k].astype(env.cfg['dtype_w']), axis = k))
		nnpu.utils.MarkScope(res_buf, 'acc')
	def proc(sc):
		if len(lhs.shape) == 1 and len(rhs.shape) == 1:
			xo, xi = sc[res].split(res.op.axis[0], factor)
			if modes == 'n':
				sc[res].tensorize(xi, env.intrins.get('VDotV', mode = 'inc'))
			elif modes == 'w':
				sc[res].tensorize(xi, env.intrins.get('VDotV', mode = modes))
		elif len(lhs.shape) == 2 and len(rhs.shape) == 2:
			xo, xi = sc[res_buf].split(res_buf.op.axis[1], gemm_shape[2])
			ko, ki = sc[res_buf].split(res_buf.op.reduce_axis[0], gemm_shape[1])
			sc[res_buf].reorder(xo, ko, res_buf.op.axis[0], xi, ki)
			if lhs.dtype == 'n':
				sc[res_buf].tensorize(xi, env.intrins.get('GEMM', shape = gemm_shape, mode = 'inc', scope_out = 'acc'))
			else:
				sc[res_buf].tensorize(xi, env.intrins.get('GEMM', shape = gemm_shape, mode = 'w', scope_out = 'acc'))
	ScheduleProcHelper.current.Add(proc)
	res = nnpu.utils.CopyAccToBuf(res_buf, 'out')
	return res