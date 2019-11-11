import nnpu
import tvm
import topi 
from nnpu.utils import ScheduleProcHelper
import numpy as np
from nnpu.intrins import IntrinManager
dtype2bytes = {'int8': 1, 'int16': 2, 'int32': 4, 'uint8': 1, 'uint16': 2, 'uint32': 4, 
               'float16': 2, 'float32': 4, 'fixed16': 2}

def dtype_bytes(dtype):
    try:
        return dtype2bytes[dtype]
    except KeyError:
        raise ValueError('unhandeled dtype: {0}'.format(dtype))
def make_intrin_call(dtype, name, *args, **kwargs):
    """ Build a llvm.NNPU intrinsic function call who has side-effect.
    Parameters
    ----------
    dtype : str
        The data type of the result. can be void to indicate no return value.
    name : str
        The name of the llvm intrinsic function 'without' llvm.NNPU prefix.
    
    num_signature : int
        I don't sure what this is, maybe used with overloaded llvm intrinsic
            function matching.
    args : list
        Poistional arguments.
    """
    name = 'llvm.NNPU.' + name
    if ('num_signature' in kwargs):
        num_signature = kwargs['num_signature']
    else:
        num_signature = 0
    return tvm.call_llvm_intrin_with_side_effect(
                    dtype, name, tvm.const(num_signature, 'uint32'), *args
                )
# def test():
#     print('aaaa')
#     env = nnpu.get_env()
#     nnpu.set_device(env)
#     shape = (64,5)
#     dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
#     a = tvm.placeholder(shape, dtype_w, name='a')
#     w = shape[0]
#     e = 16
#     def build_nms_ir(ten_in,ten_out):
#         env = nnpu.get_env()
#         ib = tvm.ir_builder.create()
#         ib.scope_attr(env.nnpu_axis, "coproc_scope", 0)
#         p_in = ib.buffer_ptr(ten_in[0])
#         p_out = ib.buffer_ptr(ten_out[0])
#         with ib.for_range(0,w, name="k") as k:
#             with ib.for_range(0,w/e, name="i") as i:
#                 ib.emit(tvm.call_extern("int32", 'NNPU_Reducebykey', ))
#             ib.emit(tvm.call_extern("int32", 'NNPU_cal_S', ))
#             with ib.for_range(0,w/e, name="j") as j:
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_max-1',))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_min-3', ))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_min3-max1',))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_GTM0', ))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_max-2', ))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_min-4', ))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_min4-max2', ))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_GTM0', ))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_(min3-max1)*(min4-max2)', ))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_cal_iou',))
#                 ib.emit(tvm.call_extern("int32", 'NNPU_comp_and_set', ))
#         stmt = ib.get()
#         return stmt
#     sph = ScheduleProcHelper()
#     a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
#     sph.MarkScope(a_buf)
#     out = tvm.extern(a_buf.shape,[a_buf],
#                     build_nms_ir,
#                     in_buffers=[tvm.decl_buffer(a_buf.shape,dtype_w,
#                                     data_alignment=8,
#                                     scope = 'local.nnpu_scratchpad')]
#                     ,out_buffers=[tvm.decl_buffer(a_buf.shape,dtype_w,
#                                     data_alignment=8,
#                                     scope = 'local.nnpu_scratchpad')]
#                     ,dtype=dtype_w,name="test_ir")
#     sph.MarkScope(out)
#     out_host, out_dram = nnpu.utils.CopyBufToH(out, 'out', sph)
#     s = tvm.create_schedule([out_host.op])
#     sph.Transform(s)
#     print(nnpu.lower(s, [a,out_host], simple_mode=True))
#     exit(0)
#     func = nnpu.build(s, [a,out_host], 'nnpu', 'llvm', name='nnpu_reduce')
#     ctx = tvm.nd.TVMContext(13, 0)
#     a_np = np.random.randint(size=(32,), dtype=a.dtype, low = 0, high = 127)
#     a_nd = tvm.nd.array(a_np, ctx)
    
#     b_nd = tvm.nd.array(np.zeros(32,).astype(out_host.dtype), ctx)

#     func(a_nd, b_nd)

#     print('a = ')
#     print(a_np)
#     print('xjb sum = ')
#     print(b_nd.asnumpy())

#     return
def test_ib():
    print('aaaa')
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (16,)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    a = tvm.placeholder(shape, dtype_w, name='a')
    w = shape[0]
    e = 16
    def build_nms_ir(ten_in,ten_out):
        ib = tvm.ir_builder.create()
        imm_value=10;
        ib.scope_attr(env.nnpu_axis, "coproc_scope", 0)
        p_in = ib.buffer_ptr(ten_in[0])
        p_out = ib.buffer_ptr(ten_out[0])
        #with ib.for_range(0,w, name="k") as k:
        with ib.for_range(0,w/e, name="i") as i:
            ib.emit(make_intrin_call("void", 'VAddI',
                        ten_out[0].access_ptr("w", 'uint32')+i*dtype_bytes(dtype_w),
                        ten_in[0].access_ptr("r", 'uint32')+i*dtype_bytes(dtype_w),
                        tvm.const(imm_value, 'float64'),
                        env.cfg['vector_unit']['size'],
                        3
                        ))
        stmt = ib.get()
        return stmt
    sph = ScheduleProcHelper()
    a_buf, a_dram = nnpu.utils.CopyHtoBuf(a, 'a', sph)
    sph.MarkScope(a_buf)
    out = tvm.extern(a_buf.shape,[a_buf],
                    build_nms_ir,
                    in_buffers=[tvm.decl_buffer(a_buf.shape,dtype_w,
                                    data_alignment=dtype_bytes(dtype_w),
                                    scope = 'local.nnpu_scratchpad0')]
                    ,out_buffers=[tvm.decl_buffer(a_buf.shape,dtype_w,
                                    data_alignment=dtype_bytes(dtype_w),
                                    scope = 'local.nnpu_scratchpad0')]
                    ,dtype=dtype_w,name="test_ir")
    sph.MarkScope(out)
    out_host, out_dram = nnpu.utils.CopyBufToH(out, 'out', sph)
    s = tvm.create_schedule([out_host.op])
    sph.Transform(s)
    print(tvm.lower(s, [a,out_host], simple_mode=True))
    print(nnpu.lower(s, [a,out_host], simple_mode=True))
    # exit(0)
    func = nnpu.build(s, [a,out_host], 'nnpu', 'llvm', name='nnpu_test')
    ctx = tvm.nd.TVMContext(13, 0)
    a_np = np.random.randint(size=(16,), dtype=a.dtype, low = 0, high = 127)
    a_nd = tvm.nd.array(a_np, ctx)
    
    b_nd = tvm.nd.array(np.zeros(16,).astype(out_host.dtype), ctx)

    func(a_nd, b_nd)

    print('a = ')
    print(a_np)
    print('xjb sum = ')
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
    test_ib()

