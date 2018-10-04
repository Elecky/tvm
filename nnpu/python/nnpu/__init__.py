from libinfo import _load_lib
from environment import Environment, get_env, init
from build_module import lower, build

# load nnpu library
_load_lib()
init()