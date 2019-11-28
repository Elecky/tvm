import tvm
import time
import numpy as np
import numpy
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import logging
import sys
import os
import topi

device = "cuda"

log_file = "cuda_vectoradd.log"
dtype = 'float32'

ctx = tvm.context(device, 0)

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
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

def vectoradd_naive():
    N = 1048576
    A = tvm.placeholder ((N,), name='A', dtype=dtype)
    B = tvm.placeholder ((N,), name='B', dtype=dtype)
    C = tvm.compute (A.shape, lambda *i: A(*i) +B(*i), name='C')
    s = tvm.create_schedule (C.op)

    bx, tx = s[C].split (C.op.axis[0], factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

    module = tvm.build(s, [A, B, C], device, target_host="llvm")

    #print(tvm.lower(s, [A, B, C], simple_mode=True))

    a = numpy.random.rand(N).astype(dtype)
    a_np = tvm.nd.array(a, ctx)
    b = numpy.random.rand(N).astype(dtype)
    b_np = tvm.nd.array(b, ctx)
    c_np = np.add(a,b)

    c_tvm = tvm.nd.array(numpy.random.rand(N).astype(dtype), ctx)
    module(a_np, b_np, c_tvm)
    
    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    evaluator = module.time_evaluator(module.entry_name, ctx, number=100)
    print(module)
    time=evaluator(a_np, b_np, c_tvm).mean
    #print('Naive: %f ms' % (evaluator(a_np, b_np, c_tvm).mean*1e3))
    print('%f GFlops'%((float(1<<20)/time/1e9)))
    # if device=="cuda":
    #     dev_module=module.imported_modules[0]
    #     print("------GPU code-----")
    #     print(dev_module.get_source())
    # else:
    #     print("error")

@autotvm.template
def vectoradd(N, dtype):
    A = tvm.placeholder ((N,), name='A', dtype=dtype)
    B = tvm.placeholder ((N,), name='B', dtype=dtype)
    C = tvm.compute (A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule (C.op)

    #schedule
    x = s[C].op.axis[0]

    ####define space begin####
    cfg = autotvm.get_config()
    #cfg.define_knob("tile_x", [32, 64, 512,1024])
    cfg.define_split("tile_x",x,num_outputs=2)
    ####define space end####

    bx,tx=cfg["tile_x"].apply(s,C,x)
    #bx,tx=s[C].split(x, cfg["tile_x"].val)

    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

    #print(tvm.lower(s, [A, B, C], simple_mode=True))

    return s, [A, B, C]

def tune_task(task,
    measure_option,
    tuner='random',
    n_trial=10,
    early_stopping=None,
    log_filename='tuning.log',
    use_transfer_learning=True):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    # create tuner
    if tuner == 'xgb' or tuner == 'xgb-rank':
        tuner_obj = XGBTuner(task, loss_type='rank')
    elif tuner == 'ga':
        tuner_obj = GATuner(task, pop_size=100)
    elif tuner == 'random':
        tuner_obj = RandomTuner(task)
    elif tuner == 'gridsearch': 
        tuner_obj = GridSearchTuner(task)
    elif tuner == 'hc':
        tuner_obj = HCTuner(task)
    elif tuner == 'lgb':
        tuner_obj = LGBTuner(task)
    elif tuner=='bs':
        tuner_obj=BSTuner(task)
    else:
        raise ValueError("Invalid tuner: " + tuner)

    if use_transfer_learning:
        if os.path.isfile(tmp_log_file):
            tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

    # do tuning
    tuner_obj.tune(n_trial=min(n_trial, len(task.config_space)),
                    early_stopping=early_stopping,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(n_trial),
                        autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune_and_evaluate(tuning_opt):
    N = 2**20
    task = autotvm.task.create(vectoradd, args=(N, 'float32'), target=device)
    print(task.config_space)

    # run tuning tasks
    print("Tuning...")
    tune_task(task, **tuning_opt)

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(log_file)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

    # print("apply history best from log file")
    # with autotvm.apply_history_best(log_file):
    #     print("Compile...")
    #     with tvm.target.create(device):
    #         s, arg_bufs = vectoradd(N, 'float32')
    #         func = tvm.build(s, arg_bufs)

    # a = numpy.random.rand(N).astype(dtype)
    # a_np = tvm.nd.array(a, ctx)
    # b = numpy.random.rand(N).astype(dtype)
    # b_np = tvm.nd.array(b, ctx)
    # c_np = a + b

    # c_tvm = tvm.nd.array(numpy.random.rand(N).astype(dtype), ctx)
    # func(a_np, b_np, c_tvm)
    
    # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    # print('Opt: %f' % evaluator(a_np, b_np, c_tvm).mean)

    print("done")

def elemwise_sum():
    x, y = 1<<5, 1<<4
    a = tvm.placeholder((x, y, y), name="a")
    b = tvm.placeholder((y, y), name="b")
    c = a + b  # same as topi.broadcast_add
    d = a * b  # same as topi.broadcast_mul
    e = topi.elemwise_sum([c, d])
    f = e / 2.0
    g = topi.sum(f)
    with tvm.target.cuda():
        sg = topi.generic.schedule_reduce(g)
        print(tvm.lower(sg, [a, b], simple_mode=True))
    func = tvm.build(sg, [a, b, g], 'cuda')
    ctx = tvm.gpu(0)
    a_np = np.random.uniform(size=(x, y, y)).astype(a.dtype)
    b_np = np.random.uniform(size=(y, y)).astype(b.dtype)
    g_np = np.sum(np.add(a_np + b_np, a_np * b_np) / 2.0)
    a_nd = tvm.nd.array(a_np, ctx)
    b_nd = tvm.nd.array(b_np, ctx)
    g_nd = tvm.nd.array(np.zeros(g_np.shape, dtype=g_np.dtype), ctx)
    func(a_nd, b_nd, g_nd)
    tvm.testing.assert_allclose(g_nd.asnumpy(), g_np, rtol=1e-5)

vectoradd_naive()
#tune_and_evaluate(tuning_option)
#elemwise_sum()
