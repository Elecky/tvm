"""Neural network operators"""
from __future__ import absolute_import

from .nn import *

get_iter_type_str = tvm.get_global_func('get_iter_type_str', True)