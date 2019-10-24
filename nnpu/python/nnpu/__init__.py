from .libinfo import _load_lib
from .environment import Environment, get_env, set_device, set_dump, set_profile
from .build_module import lower, build, build_config
from .utils import *
from .utils import create_schedule
import os

from . import top
# load nnpu library
_load_lib()

func = tvm.get_global_func('nnpu.patch_irprinter', False)
func()

func = tvm.get_global_func('nnpu.init_uop_template', False)
func(os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(__file__))), '..', '..', 'config', 'micro_code_template.txt'))