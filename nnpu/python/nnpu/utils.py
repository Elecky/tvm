import tvm
from .environment import get_env
from .helper import convert_scope
import topi

class ScheduleProcHelper(object):
    '''
    a helper method to collect the schedule transforming closures.
    '''
    current = None

    def __init__(self):
        self.closures = []
        self.env = get_env()
        pass

    def Add(self, closure):
        self.closures.append(closure)
        pass

    def Transform(self, sc):
        for f in self.closures:
            f(sc)
        self.closures = []
    
    def MarkScope(self, tensor, scope='uni'):
        scope = convert_scope(self.env, scope, include_acc=True)
        #print('marking scope:')
        #print(scope)
        self.Add(lambda sc: sc[tensor].set_scope(scope))
    
    def __enter__(self):
        self.last = ScheduleProcHelper.current
        ScheduleProcHelper.current = self
        return self
    
    def __exit__(self, ptype, value, trace):
        ScheduleProcHelper.current = self.last

def MarkScope(tensor, scope='uni', sph=None):
    if (sph):
        sph.MarkScope(tensor, sph)
    else:
        ScheduleProcHelper.current.MarkScope(tensor, scope)

def DMACopyHtoDram(tensor, name_prefix, sph=None):
    sph = ScheduleProcHelper.current if sph is None else sph
    
    env = get_env()
    tensor_dram = tvm.compute(tensor.shape, lambda *i: tensor(*i), name_prefix + "_dram")
    
    sph.Add(lambda sc: sc[tensor_dram].set_scope(env.dram_scope))
    sph.Add(lambda sc: sc[tensor_dram].pragma(sc[tensor_dram].op.axis[0], env.dma_copy_pragma))
    
    return tensor_dram

def CopyHtoBuf(tensor, name_prefix, sph=None, dst_scope='uni'):
    sph = ScheduleProcHelper.current if sph is None else sph

    env = get_env()
    tensor_dram = DMACopyHtoDram(tensor, name_prefix, sph)
    tensor_buf = tvm.compute(tensor.shape, lambda *i: tensor_dram(*i), name_prefix + "_buf")
    
    scope = convert_scope(env, dst_scope)
    sph.Add(lambda sc: sc[tensor_buf].set_scope(scope))
    sph.Add(lambda sc: sc[tensor_buf].pragma(sc[tensor_buf].op.axis[0], env.scratchpad_ls))

    return tensor_buf, tensor_dram

def CopyBufToDram(tensor, name_prefix, sph=None):
    sph = ScheduleProcHelper.current if sph is None else sph

    env = get_env()
    tensor_dram = tvm.compute(tensor.shape, lambda *i: tensor(*i), name_prefix + "_dram")
    
    sph.Add(lambda sc: sc[tensor_dram].set_scope(env.dram_scope))
    sph.Add(lambda sc: sc[tensor_dram].pragma(sc[tensor_dram].op.axis[0], env.scratchpad_ls))
    
    return tensor_dram

def CopyBufToH(tensor, name_prefix, sph=None):
    sph = ScheduleProcHelper.current if sph is None else sph

    env = get_env()
    tensor_dram = CopyBufToDram(tensor, name_prefix, sph)
    tensor_host = tvm.compute(tensor_dram.shape, lambda *i: tensor_dram(*i), name_prefix + '_host')
    
    sph.Add(lambda sc: sc[tensor_host].pragma(sc[tensor_host].op.axis[0], env.dma_copy_pragma))

    return tensor_host, tensor_dram

def PragmaCopy(tensor, sph=None):
    env = get_env()
    sph = ScheduleProcHelper.current if sph is None else sph
    sph.Add(lambda sc: sc[tensor].pragma(tensor.op.axis[0], env.scratchpad_copy))

def reshape(tensor, shape, sph=None, dst_scope='uni'):
    res = topi.reshape(tensor, shape)
    
    MarkScope(res, dst_scope)
    PragmaCopy(res)

    return res

def transpose(tensor, axes=None, sph=None, dst_scope='uni'):
    res = topi.transpose(tensor, axes)

    MarkScope(res, dst_scope)
    PragmaCopy(res)
    return res

def CopyAcc2Buf(tensor, name, dst_scope='uni', sph=None):
    res = tvm.compute(tensor.shape, lambda *i: tensor(*i), name)
    MarkScope(res, dst_scope, sph)
    env = get_env()
    sph = ScheduleProcHelper.current if sph is None else sph
    sph.Add(lambda sc: sc[res].pragma(res.op.axis[0], env.copy_acc2buf))

    return res

def create_schedule(*args, **kwargs):
    s = tvm.create_schedule(*args, **kwargs)
    ScheduleProcHelper.current.Transform(s)
    return s