from .libinfo import _load_lib
from .environment import Environment, get_env, set_device, set_dump
from .build_module import lower, build
from .utils import *
from .utils import create_schedule

# load nnpu library
_load_lib()