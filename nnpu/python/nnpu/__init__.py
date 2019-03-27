from .libinfo import _load_lib
from .environment import Environment, get_env, set_device, set_dump
from .build_module import lower, build, build_config
from .utils import *
from .utils import create_schedule

from .ir_pass import lift_coproc_scope
from . import top
# load nnpu library
_load_lib()