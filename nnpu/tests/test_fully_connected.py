import nnpu
import tvm
import topi
from nnpu.utils import ScheduleProcHelper
import numpy as np

env = nnpu.get_env()
nnpu.set_device(env)