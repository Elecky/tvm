import nnpu
import tvm
import topi 
from nnpu.utils import ScheduleProcHelper
import numpy as np
from nnpu.intrins import IntrinManager


def test():
    print('aaaa')
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (32,)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder(shape, dtype_w, name='a')
    def build_nms_ir(ten_in,ten_out):
        env = nnpu.get_env()
        ib = tvm.ir_builder.create()
        ib.scope_attr(env.nnpu_axis, "coproc_scope", 0)
        p_in = ib.buffer_ptr(ten_in[0])
        p_out = ib.buffer_ptr(ten_out[0])
        '''
        ib.emit(tvm.call_extern("int32", 'NNPU_Memset',
                                ten_out[0].access_ptr('w', 'uint32'), 32, 2,
                                str(0), 3
                    ))
        '''
        five = tvm.const(5, env.cfg['dtype_w'])
        with ib.for_range(0, 32, name="i") as i:
            p_out[i] = p_in[i]
            with ib.if_scope((i % 2) == 0):
                p_out[i] = p_in[i] + five
        stmt = ib.get()
        return stmt
    #sph = ScheduleProcHelper()
    #a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    #sph.MarkScope(a_buf)
    b_buf = tvm.extern([(32,)],[a], build_nms_ir ,dtype=dtype_w,tag="test_ir")
    #sph.MarkScope(b_buf)
    #b_host, b_dram = nnpu.utils.CopyBufToH(b_buf, 'b', sph)
    s = tvm.create_schedule([b_buf.op])
    #sph.Transform(s)
    print(nnpu.lower(s, [a ,b_buf], simple_mode=True))
    exit(0)
    func = nnpu.build(s, [a ,b_host], 'nnpu', 'llvm', name='nnpu_reduce')
    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(5,16), dtype=a.dtype, low = 0, high = 127)
    a_nd = tvm.nd.array(a_np, ctx)
    
    b_nd = tvm.nd.array(np.zeros((5,)).astype(b_host.dtype), ctx)

    func(a_nd, b_nd)

    print('a = ')
    print(a_np)
    print('reduce sum row = ')
    print(b_nd.asnumpy())

    return

'''
def test():
    print('aaaa')
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (5,32)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder(shape, dtype_w, name='a')

    def fcombine(x, y):
        lhs = tvm.select((x[1] >= y[1]), x[0], y[0])
        rhs = tvm.select((x[1] >= y[1]), x[0], y[0])
        return lhs,rhs

    def fidentity(t0, t1):
        return tvm.const(0,dtype_w), tvm.const(0,dtype_w)

    argmax = tvm.comm_reducer(fcombine, fidentity, name='argmax')
    sph = ScheduleProcHelper()

    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    k = tvm.reduce_axis((0, 32), 'k')

    b_buf , _ = tvm.compute((5,),lambda i: argmax((a_buf[i,k], a_buf[4, k]), axis=k), 'VReduceKey')
    
    sph.MarkScope(b_buf)
    b_host, b_dram = nnpu.utils.CopyBufToH(b_buf, 'b', sph)
    X02 = b_buf[0]
    NX = tvm.compute((32,),lambda i: tvm.select(a_buf[0,i] > X02, a_buf[0,i], X02), 'NX')    
    sph.MarkScope(NX)
    NX_host, NX_dram = nnpu.utils.CopyBufToH(NX, 'NX', sph)
    s = tvm.create_schedule([NX_host.op,b_host.op])
    sph.Transform(s)
    ko, ki = s[NX].split(NX.op.axis[0], factor=16)
    s[NX].tensorize(ki, env.intrins.get('VGTMS', mode='w'))
    ko, ki = s[b_buf].split(b_buf.op.reduce_axis[0], factor=16)
    s[b_buf].reorder( ko,b_buf.op.axis[0],ki)
    #s[b_buf].tensorize(b_buf.op.axis[0], env.intrins.get('VReduceKey', mode='w'))
    #ko, ki = s[b_buf].split(b_buf.op.reduce_axis[0], factor=1)
    #xo,xi = s[b_buf].split(b_buf.op.axis[0], factor=1)
    #s[b_buf].reorder( b_buf.op.reduce_axis[0],b_buf.op.axis[0])
    #s[b_buf].tensorize(b_buf.op.axis[0], env.intrins.get('VReduceKey', mode='w'))
    print(nnpu.lower(s, [a ,b_host], simple_mode=True))
    exit(0)
    func = nnpu.build(s, [a ,b_host], 'nnpu', 'llvm', name='nnpu_reduce')
    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(5,16), dtype=a.dtype, low = 0, high = 127)
    a_nd = tvm.nd.array(a_np, ctx)
    
    b_nd = tvm.nd.array(np.zeros((5,)).astype(b_host.dtype), ctx)

    func(a_nd, b_nd)

    print('a = ')
    print(a_np)
    print('reduce sum row = ')
    print(b_nd.asnumpy())

    return
'''
if __name__ == '__main__':
    test()

