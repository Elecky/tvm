"Example code to perform int8 GEMM"
import logging
import sys
import numpy as np
import tvm
from tvm import autotvm
from topi.cuda.tensor_intrin import dp4a

DO_TUNING = True
PRETUNED_INDEX = 75333

intrin_dp4a = dp4a('local', 'local', 'local')

@autotvm.template
def gemm_float32():
    # graph
    nn = 2048
    n = tvm.var('n')
    n = tvm.convert(nn)
    m, l = n, n
    A = tvm.placeholder((l, n), name='A')
    B = tvm.placeholder((l, m), name='B')
    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute(
        (m, n),
        lambda ii, jj: tvm.sum(A[k, jj] * B[k, ii], axis=k),
        name='C')

    # schedule
    s = tvm.create_schedule(C.op)
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    #get the GPU thread indices
    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, 2), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, 2), "vthread", name="vy")

    y,x=s[C].op.axis
    cfg=autotvm.get_config()
    cfg.define_split("tile_y",y,num_outputs=2)
    cfg.define_split("tile_x",x,num_outputs=2)

    #split the workloads
    # by, yi = s[C].split(C.op.axis[0], factor=block_factor)
    # bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
    # by,yi=cfg["tile_y"].apply(s,C,C.op.axis[0])
    # bx,xi=cfg["tile_x"].apply(s,C,C.op.axis[1])
    by,yi=s[C].split(y,factor=cfg["tile_y"].size[1])
    bx,xi=s[C].split(x,factor=cfg["tile_x"].size[1])
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].reorder(by, bx, yi, xi)

    tyz, yi = s[C].split(yi, nparts=2)
    ty, yi = s[C].split(yi, nparts=num_thread)
    txz, xi = s[C].split(xi, nparts=2)
    tx, xi = s[C].split(xi, nparts=num_thread)
    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(tyz, txz, ty, tx, yi, xi)
    s[CC].compute_at(s[C], tx)

    yo, xo = CC.op.axis
    ko, ki = s[CC].split(k, factor=8)
    kt, ki = s[CC].split(ki, factor=1)
    s[CC].reorder(ko, kt, ki, yo, xo)
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[CC].unroll(kt)
    s[AL].compute_at(s[CC], kt)
    s[BL].compute_at(s[CC], kt)
    # Schedule for A's shared memory load
    ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thread)
    _, xi = s[AA].split(s[AA].op.axis[1], factor=num_thread * 4)
    tx, xi = s[AA].split(xi, nparts=num_thread)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(xi)
    # Schedule for B' shared memory load
    ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thread)
    _, xi = s[BB].split(s[BB].op.axis[1], factor=num_thread * 4)
    tx, xi = s[BB].split(xi, nparts=num_thread)
    s[BB].bind(ty, thread_y)
    s[BB].bind(tx, thread_x)
    s[BB].vectorize(xi)

    return s, [A, B, C]


if __name__ == '__main__':
    task = autotvm.task.create(gemm_float32, args=(), target="cuda")
    print(task.config_space)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )

    log_name = 'gemm_float32.log'
    if DO_TUNING:
        tuner = autotvm.tuner.LGBTuner(task)
        prefix = "[Task %2d/%2d] " %(1, 1)
        tuner.tune(n_trial=1000, measure_option=measure_option,
                   callbacks=[
                       autotvm.callback.progress_bar(1000, prefix=prefix),
                       autotvm.callback.log_to_file(log_name)])

        dispatch_context = autotvm.apply_history_best(log_name)
        best_config = dispatch_context.query(task.target, task.workload)
        print('\nBest config:')
        print(best_config)
    else:
        config = task.config_space.get(PRETUNED_INDEX)
        dispatch_context = autotvm.task.ApplyConfig(config)
        print("Using pretuned config:")
        print(config)
        
    n=m=l=2048
    with dispatch_context:
        with tvm.target.create('cuda'):
            s, arg_bufs = gemm_float32()
            f = tvm.build(s, arg_bufs, 'cuda', name='gemm_float32')

    ctx = tvm.context('cuda', 0)
    a_np = np.random.uniform(size=(n, l)).astype("float32")
    b_np = np.random.uniform(size=(m, l)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((n, m), dtype="float32"), ctx)
    for i in range(2):
        f(a, b, c)
    tvm.testing.assert_allclose(
        c.asnumpy(), np.dot(b_np.T, a_np), rtol=1e-5)

    num_ops = 2 * l * m * n
    num_runs = 10
    timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GOPS = num_ops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GOPS." %
          (num_runs, t * 1e3, GOPS))