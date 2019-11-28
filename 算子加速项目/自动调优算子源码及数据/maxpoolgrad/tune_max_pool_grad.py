# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Writing tunable template and Using auto-tuner
=============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_

This is an introduction tutorial to the auto-tuning module in TVM.

There are two steps in auto-tuning.
The first step is defining a search space.
The second step is running a search algorithm to explore through this space.
In this tutorial, you can learn how to perform these two steps in TVM.
The whole workflow is illustrated by a matrix multiplication example.
"""

######################################################################
# Install dependencies
# --------------------
# To use autotvm package in TVM, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost
#
# To make TVM run faster in tuning, it is recommended to use cython
# as FFI of TVM. In the root directory of TVM, execute
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import logging
import sys

import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner,HCTuner,LGBTuner
import topi
import topi.testing
import math
from topi.util import get_const_tuple
######################################################################
# Step 1:  Define the search space
# --------------------------------
# In this section, we will rewrite a deterministic TVM schedule code to a
# tunable schedule template. You can regard the process of search space definition
# as the parameterization of our existing schedule code.
#
# To begin with, here is how we implement a blocked matrix multiplication in TVM.

# Matmul V0: Constant tiling factor
def matmul_v0(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    yo, yi = s[C].split(y, 8)
    xo, xi = s[C].split(x, 8)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]

#####################################################################
# Parametrize the schedule
# ^^^^^^^^^^^^^^^^^^^^^^^^
# In the previous schedule code, we use a constant "8" as tiling factor.
# However, it might not be the best one because the best tiling factor depends
# on real hardware environment and input shape.
#
# If you want the schedule code to be portable across a wider range of input shapes
# and target hardware, it is better to define a set of candidate values and
# pick the best one according to the measurement results on target hardware.
#
# In autotvm, we can define a tunable parameter, or a "knob" for such kind of value.

# Matmul V1: List candidate values
@autotvm.template  # 1. use a decorator
def matmul_v1(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # 2. get the config object
    cfg = autotvm.get_config()

    # 3. define search space
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16])

    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg['tile_y'].val)
    xo, xi = s[C].split(x, cfg['tile_x'].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]

###############################################################################
# Here we make four modifications to the previous schedule code and get
# a tunable "template". We can explain the modifications one by one.
#
# 1. Use a decorator to mark this function as a simple template.
# 2. Get a config object:
#    You can regard this :code:`cfg` as an argument of this function but
#    we obtain it in a different way. With this argument, this function is no longer
#    a deterministic schedule code. Instead, we can pass different configurations to
#    this function and get different schedules, so this function is a "template".
#
#    To make the template function more compact, we do two things in a single function.
#    (1) define a search space and (2) schedule according to an entity in this space.
#    To achieve this, we make :code:`cfg` be either
#    a :any:`ConfigSpace` or a :any:`ConfigEntity` object.
#
#    When it is a :any:`ConfigSpace`, it will collect all tunable knobs in this function and
#    build the search space.
#    When it is a :any:`ConfigEntity`, it will ignore all space definition API
#    (namely, :code:`cfg.define_XXXXX(...)`).   Instead, it stores deterministic values for
#    all tunable knobs, and we schedule according to these values.
#
#    During auto-tuning, we will first call this template with a :any:`ConfigSpace`
#    object to build the search space. Then we call this template with different :any:`ConfigEntity`
#    in the built space to get different schedules. Finally we will measure the code generated by
#    different schedules and pick the best one.
#
# 3. Define two tunable knobs. The first one is :code:`tile_y` with
#    5 possible values. The second one is :code:`tile_x` with a same
#    list of possible values. These two knobs are independent, so they
#    span a search space with size = 5x5 = 25
# 4. Schedule according to the deterministic values in :code:`cfg`
#

#####################################################################
# Use better space definition API
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the previous template, we manually list all possible values for a knob.
# This is the lowest level API to define the space.
# However, we also provide another set of API to make the space definition
# easier and smarter. It is recommended to use this set of high level API.
#
# In the flowing example, we use :any:`ConfigSpace.define_split` to define a split
# knob. It will enumerate all the possible ways to split an axis and construct
# the space.
#
# We also have :any:`ConfigSpace.define_reorder` for reorder knob and
# :any:`ConfigSpace.define_annotate` for annotation like unroll, vectorization,
# thread binding.
# When the high level API cannot meet your requirement, you can always fall
# back to use low level API.

@autotvm.template
def matmul(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    # schedule
    y, x = s[C].op.axis
    print(y)
    print(x)
    k = s[C].op.reduce_axis[0]
    print(k)

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]

@autotvm.template
def verify_pool_grad():
    n  = 1 
    ic = 256 
    ih = 32
    kh = 3
    sh = 2
    padding = [0,0,0,0]
    pool_type = 'max'
    ceil_mode = False
    count_include_pad=True
    add_relu=False

    iw = ih
    kw = kh
    sw = sh
    pt, pl, pb, pr = padding
    layout = "NCHW"
    A = tvm.placeholder((n, ic, ih, iw), name='A')
    B = topi.nn.pool(A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                     pool_type=pool_type, ceil_mode=ceil_mode,
                     layout="NCHW", count_include_pad=count_include_pad)
    dtype = A.dtype

    bshape = get_const_tuple(B.shape)
    ashape = get_const_tuple(A.shape)
    if ceil_mode:
        assert bshape[2] == int(math.ceil(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.ceil(float(ashape[3] - kw + pl + pr) / sw) + 1)
    else:
        assert bshape[2] == int(math.floor(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.floor(float(ashape[3] - kw + pl + pr) / sw) + 1)
    OutGrad = tvm.placeholder(bshape, name='OutGrad')
    PoolGrad = topi.nn.pool_grad(OutGrad, A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                                 pool_type=pool_type, ceil_mode=ceil_mode,
                                 layout="NCHW", count_include_pad=count_include_pad)

    a_np = np.random.uniform(low=0.001, size=(n, ic, ih, iw)).astype(dtype)
    out_grad_np = np.random.uniform(low=0.001, size=bshape).astype(dtype)
    
    ctx = tvm.context('cuda', 0)
    s = tvm.create_schedule(PoolGrad.op)


    # 加载到共享内存 并有各自的线程加载
    max_pool_index = PoolGrad.op.input_tensors[0]    
    AShared = s.cache_read(A, 'shared', [max_pool_index])
    ASharedLocal = s.cache_read(AShared, "local", [max_pool_index])
    OutGradShared= s.cache_read(OutGrad, "shared", [PoolGrad])
    OutGradSharedLocal = s.cache_read(OutGradShared, "local", [PoolGrad])
    # PoolGradLocal = s.cache_write(PoolGrad, "local")
     
    # prepare blocks and threads
   
 

    
    
    num_thread = 1024
    step = 8
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis("threadIdx.x")
    # thread_y = tvm.thread_axis("threadIdx.y")

    axisbatch,axischanel,axisheight,axiswidth = s[PoolGrad].op.axis
    fused = s[PoolGrad].fuse(axisheight,axiswidth )
     
    # get the config object
    cfg = autotvm.get_config()
    cfg.add_flop(4*n*ic*ih*ih)
    # cfg.define_split("tile_x",fused, num_outputs=2)
    # define search space
    li = [lambda : [np.power(2,x) for x in range(0,12)]]
    cfg.define_knob("tile_x", li[0]())
    # schedule according to config
    bx, tx = s[PoolGrad].split(fused, cfg['tile_x'].val)
    # bx, tx = cfg["tile_x"].apply(s, PoolGrad, fused)

    # by, ty = s[PoolGrad].split(axischanel, factor=num_thread)
    s[PoolGrad].bind(axisbatch,block_z)
    s[PoolGrad].bind(axischanel,block_y)
    # s[PoolGrad].bind(ty,thread_y)
    s[PoolGrad].bind(bx,block_x)
    s[PoolGrad].bind(tx,thread_x)

    s[max_pool_index].compute_at(s[PoolGrad], tx)
    
    ##  schedule for share memory in PoolGrad
    s[AShared].compute_at(s[PoolGrad], tx)
    s[ASharedLocal].compute_at(s[PoolGrad], tx)
    s[OutGradShared].compute_at(s[PoolGrad], tx)
    s[OutGradSharedLocal].compute_at(s[PoolGrad], tx)
## schedule for ashared
    yi, xi, ci, ni = s[AShared].op.axis
    Afused = s[AShared].fuse(yi, xi, ci, ni)
    # xo,xi = s[AShared].split(Afused,1024)
    xo,xi = s[AShared].split(Afused, cfg['tile_x'].val)
    s[AShared].bind(xi, thread_x)

## schedule for outputgrad
    yi, xi, ci, ni = s[OutGradShared].op.axis
    Outfused = s[OutGradShared].fuse(yi, xi, ci, ni)
    # xo, xi = s[OutGradShared].split(Outfused, 1024)
    xo,xi = s[OutGradShared].split(Outfused, cfg['tile_x'].val)
    s[OutGradShared].bind(xi, thread_x)    
   
    return s, [A, OutGrad, PoolGrad]


@autotvm.template
def verify_pool_grad_noshare():
                    

    n  = 1 
    ic = 256
    ih = 32
    kh = 3
    sh = 2
    padding = [0,0,0,0]
    pool_type = 'max'
    ceil_mode = False
    count_include_pad=True
    add_relu=False
    iw = ih
    kw = kh
    sw = sh
    pt, pl, pb, pr = padding
    layout = "NCHW"
    A = tvm.placeholder((n, ic, ih, iw), name='A')
    B = topi.nn.pool(A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                     pool_type=pool_type, ceil_mode=ceil_mode,
                     layout="NCHW", count_include_pad=count_include_pad)
    dtype = A.dtype

    bshape = get_const_tuple(B.shape)
    ashape = get_const_tuple(A.shape)
    if ceil_mode:
        assert bshape[2] == int(math.ceil(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.ceil(float(ashape[3] - kw + pl + pr) / sw) + 1)
    else:
        assert bshape[2] == int(math.floor(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.floor(float(ashape[3] - kw + pl + pr) / sw) + 1)
    OutGrad = tvm.placeholder(bshape, name='OutGrad')
    PoolGrad = topi.nn.pool_grad(OutGrad, A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                                 pool_type=pool_type, ceil_mode=ceil_mode,
                                 layout="NCHW", count_include_pad=count_include_pad)

    a_np = np.random.uniform(low=0.001, size=(n, ic, ih, iw)).astype(dtype)
    out_grad_np = np.random.uniform(low=0.001, size=bshape).astype(dtype)
    
    ctx = tvm.context('cuda', 0)
    s = tvm.create_schedule(PoolGrad.op)

    

    
    # prepare blocks and threads

    # 1  get config
    cfg = autotvm.get_config()
    cfg.add_flop(4*n*ic*ih*ih)
    
    # define search space

    li = [lambda : [np.power(2,x) for x in range(0,12)]]
    cfg.define_knob("tile_x", li[0]())
    # schedule according to config
    

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")
    # thread_y = tvm.thread_axis((0, num_thread),"threadIdx.y")

  
    fused = s[PoolGrad].fuse( * s[PoolGrad].op.axis )
    # 2 define search scope
    # cfg.define_split("tile_x",fused, num_outputs=2)
    
    # bx, tx = s[PoolGrad].split(fused, factor=num_thread)
    bx, tx = s[PoolGrad].split(fused, cfg['tile_x'].val)

    # 3 schedule of conifg
    # bx, tx = cfg["tile_x"].apply(s,PoolGrad,fused)
    # by, ty = s[PoolGrad].split(axischanel, factor=num_thread)
    
    # s[PoolGrad].bind(ty,thread_y)
    s[PoolGrad].bind(bx,block_x)
    s[PoolGrad].bind(tx,thread_x)

    max_pool_index = PoolGrad.op.input_tensors[0]

    # print(PoolGrad.op.input_tensors)
   

    s[max_pool_index].compute_at(s[PoolGrad], tx)
    return s, [A, OutGrad, PoolGrad]
@autotvm.template
def verify_argmax():
    n  = 1 
    ic = 256
    ih = 32
    kh = 3
    sh = 2
    padding = [0,0,0,0]
    pool_type = 'max'
    ceil_mode = False
    count_include_pad=True
    add_relu=False
    iw = ih
    kw = kh
    sw = sh
    pt, pl, pb, pr = padding
    layout = "NCHW"
    A = tvm.placeholder((n, ic, ih, iw), name='A')
    B = topi.nn.pool(A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                     pool_type=pool_type, ceil_mode=ceil_mode,
                     layout="NCHW", count_include_pad=count_include_pad)
    dtype = A.dtype

    bshape = get_const_tuple(B.shape)
    ashape = get_const_tuple(A.shape)
    if ceil_mode:
        assert bshape[2] == int(math.ceil(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.ceil(float(ashape[3] - kw + pl + pr) / sw) + 1)
    else:
        assert bshape[2] == int(math.floor(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.floor(float(ashape[3] - kw + pl + pr) / sw) + 1)
    OutGrad = tvm.placeholder(bshape, name='OutGrad')
    PoolGrad = topi.nn.pool_grad(OutGrad, A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                                 pool_type=pool_type, ceil_mode=ceil_mode,
                                 layout="NCHW", count_include_pad=count_include_pad)

    a_np = np.random.uniform(low=0.001, size=(n, ic, ih, iw)).astype(np.float32)
    out_grad_np = np.random.uniform(low=0.001, size=bshape).astype(dtype)
    ctx = tvm.context('cuda', 0)
    max_pool_index = PoolGrad.op.input_tensors[0]
    s = tvm.create_schedule(max_pool_index.op)
    # prepare blocks and threads
    num_thread = 1024


    # 1  get config
    cfg = autotvm.get_config()
    cfg.add_flop(4*n*ic*ih*ih)
    # 2 define search scope
    # cfg.define_split("tile_x",fused, num_outputs=2)
    li = [lambda : [np.power(2,x) for x in range(0,12)]]
    cfg.define_knob("tile_x", li[0]())
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")
    fused = s[max_pool_index].fuse(*s[max_pool_index].op.axis )
    # bx, tx = s[max_pool_index].split(fused, factor=num_thread)
    bx, tx = s[max_pool_index].split(fused, cfg['tile_x'].val)
    s[max_pool_index].bind(bx,block_x)
    s[max_pool_index].bind(tx,thread_x)
    print(tvm.lower(s, [A, OutGrad , max_pool_index], simple_mode=True))

    return s, [A, OutGrad, PoolGrad]

######################################################################
# .. note:: More Explanation on :code:`cfg.defile_split`
#
#  In this template, :code:`cfg.define_split("tile_y", y, num_outputs=2)` will enumerate
#  all possible combinations that can split axis y into two axes with factors of the length of y.
#  For example, if the length of y is 32 and we want to split it into two axes
#  using factors of 32, then there are 6 possible values for
#  (length of outer axis, length of inner axis) pair, namely
#  (32, 1), (16, 2), (8, 4), (4, 8), (2, 16) or (1, 32).
#  They are just the 6 possible values of `tile_y`.
#
#  During schedule, :code:`cfg["tile_y"]` is a :code:`SplitEntity` object.
#  We stores the lengths of outer axes and inner axes in :code:`cfg['tile_y'].size`
#  (a tuple with two elements).
#  In this template, we apply it by using :code:`yo, yi = cfg['tile_y'].apply(s, C, y)`.
#  Actually, this is equivalent to
#  :code:`yo, yi = s[C].split(y, cfg["tile_y"].size[1])`
#  or  :code:`yo, yi = s[C].split(y, nparts=cfg['tile_y"].size[0])`
#
#  The advantage of using cfg.apply API is that it makes multi-level split
#  (when num_outputs >= 3) easier.

######################################################################
# Step 2:  Search through the space
# ---------------------------------
# In step 1, we build the search space by extending our old schedule code
# into a template. The next step is to pick a tuner and explore in this space.
#
# Auto-tuners in TVM
# ^^^^^^^^^^^^^^^^^^
# The job for a tuner can be described by following pseudo code
#
#   .. code-block:: c
#
#    ct = 0
#    while ct < max_number_of_trials:
#        propose a batch of configs
#        measure this batch of configs on real hardware and get results
#        ct += batch_size
#
# When proposing the next batch of configs, the tuner can take different strategies. We
# provide four tuners with different strategies in autotvm.
#
# * :any:`RandomTuner`: Enumerate the space in a random order
# * :any:`GridSearchTuner`: Enumerate the space in a grid search order
# * :any:`GATuner`: Using genetic algorithm to search through the space
# * :any:`XGBTuner`: Uses a model based method. Train a XGBoost model to predict the speed of lowered IR and pick the next batch according to the prediction.
#
# You can choose the tuner according to the size of your space, your time budget and other factors.
# For example, if your space is very small (less than 1000), a gridsearch tuner or a
# random tuner is good enough. If your space is at the level of 10^9 (this is the space
# size of a conv2d operator on CUDA GPU), XGBoostTuner can explore more efficiently
# and find better configs.

################################################################
# Begin tuning
# ^^^^^^^^^^^^
# Here we continue our matrix multiplication example.
# First we should create a tuning task.
# We can also inspect the initialized search space.
# In this case, for a 512x512 square matrix multiplication, the space size
# is 10x10=100
# N, L, M = 512, 512, 512
# task = autotvm.task.create(matmul, args=(N, L, M, 'float32'), target='llvm')
# print(task.config_space)

# task=autotvm.task.create(verify_argmax,args='',target='cuda')
# task=autotvm.task.create(verify_pool_grad_noshare,args='',target='cuda')
task=autotvm.task.create(verify_pool_grad,args='',target='cuda')
print(task.config_space)

################################################################
# Then we need to define how to measure the generated code and pick a tuner.
# Since our space is small, a random tuner is just okay.
#
# We only make 10 trials in this tutorial for demonstration. In practice,
# you can do more trials according to your time budget.
# We will log the tuning results into a log file. This file can be
# used to get the best config later.

# logging config (for printing tuning log to the screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure 5 times and take average to reduce variance.
measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

# begin tuning, log records to file `matmul.log`
#  XGBTuner, GATuner, RandomTuner, GridSearchTuner,HCTuner,LGBTuner

list = [LGBTuner]
import time 
for tuneAlgrm in list:
    tuner = tuneAlgrm(task)
    start = time.time()
    tuner.tune(n_trial=2000,
 early_stopping=400,
           measure_option=measure_option)
    end = time.time()
    print(tuner)
    print('运行时间为  ' + str(end - start))
#########################################################################
# Finally we apply history best from the cache file and check its correctness.
# We can call the function :code:`matmul` directly under the
# :any:`autotvm.apply_history_best` context. When we call this function,
# it will query the dispatch context with its argument and get the best config
# with the same argument.

# apply history best from log file
# with autotvm.apply_history_best('matmul.log'):
#     with tvm.target.create("llvm"):
#         s, arg_bufs = matmul(N, L, M, 'float32')
#         func = tvm.build(s, arg_bufs)

# # check correctness
# a_np = np.random.uniform(size=(N, L)).astype(np.float32)
# b_np = np.random.uniform(size=(L, M)).astype(np.float32)
# c_np = a_np.dot(b_np)

# c_tvm = tvm.nd.empty(c_np.shape)
# func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

# tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
