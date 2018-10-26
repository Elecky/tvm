import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np


def test():
    env = nnpu.get_env()
    nnpu.set_device(env)
    shape = (16,)
    bigshape =(128,)
    dtype_n, dtype_w = env.cfg['dtype_n'], env.cfg['dtype_w']
    assert bigshape[0] % shape[0] == 0, 'the big vctr size is wrong' 
    a = tvm.placeholder(bigshape, dtype_w, 'a')
    b = tvm.placeholder(bigshape, dtype_w, 'b')
    n_sheet=bigshape[0]//shape[0]

if __name__ == '__main__':
    test()