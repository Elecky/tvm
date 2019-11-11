'''the environment and context of NNPU hardware simulator
some code are from tvm/vta
'''

import os
import json
import copy
import tvm
import yaml

from .intrins import IntrinManager
from .intrins import make_intrin_call
from .helper import convert_scope

class Environment(object):
    """Hardware configuration object.

    This object contains all the information
    needed for compiling to a specific NNPU backend.

    Parameters
    ----------
    cfg : dict of str to value.
        The configuration parameters.

    Example
    --------
    .. code-block:: python

      # the following code reconfigures the environment
      # temporarily to attributes specified in new_cfg.json
      new_cfg = json.load(json.load(open("new_cfg.json")))
      with vta.Environment(new_cfg):
          # env works on the new environment
          env = vta.get_env()
    """

    current = None
    # some constants
    dram_scope = 'local.nnpu_dram'
    scratchpad_scope_prefix = 'local.nnpu_scratchpad'
    uni_scratchpad_scope = scratchpad_scope_prefix + '0'
    acc_scope = 'local.nnpu_acc_buffer'
    # compiler pragmas
    dma_copy_pragma = 'nnpu_dma_copy'
    dma_copy_to_buf = 'nnpu_dma_to_scratchpad'
    dma_copy_from_buf = dma_copy_to_buf
    scratchpad_ls = 'nnpu_scratchpad_ls'
    scratchpad_copy = 'nnpu_scratchpad_copy'
    copy_acc2buf = 'nnpu_copy_acc2buf'
    # pipeline IDs:
    pid_dma_copy = 1
    pid_matrix_compute = 2
    pid_acc2buf_copy = 2
    pid_vector_compute = 3
    def pid_scratchpad_copy(self, dst):
        return 4 + self.scratchpad_scope_to_idx(dst)

    def __init__(self, cfg_path):
        self.cfg = {}

        cfg = yaml.load(open(cfg_path), Loader=yaml.SafeLoader)
        self.cfg_path = cfg_path
        self.cfg.update(cfg)

        self.nnpu_axis = tvm.thread_axis('nnpu')

        self.intrins = IntrinManager(self)
        pass

    def __enter__(self):
        self.last_env = Environment.current
        Environment.current = self
        #set_device(self)
        return self

    def __exit__(self, ptype, value, trace):
        Environment.current = self.last_env
        # reset device based on the last Environment
        #set_device(Environment.current)

    def scratchpad_scope(self, scratchpad_id = 0):
        assert scratchpad_id < 8, 'scratchpad_id should be less than 8'
        return self.scratchpad_scope_prefix + str(scratchpad_id)

    def get_scope(self, name):
        return convert_scope(self, name, True)

    def scratchpad_scope_to_idx(self, scope):
        assert self.is_scratchpad_scope(scope), \
            '{0} is not a scratchpad scope name'
        id = scope[len(self.scratchpad_scope_prefix):]
        id = int(id)
        assert id < 8, 'scratchpad_id should be less than 8'
        return id

    def is_scratchpad_scope(self, scope):
        return scope.startswith(self.scratchpad_scope_prefix)

    def get_scope_config(self, scope):
        key = None
        if (self.is_scratchpad_scope(scope)):
            id = self.scratchpad_scope_to_idx(scope)
            key = 'scratchpad' + str(id)
        elif (scope == self.acc_scope):
            key = 'acc_buffer'
        else:
            raise ValueError('illegal scope name')
        return self.cfg[key]

    def get_pid(self, pid):
        return pid

# set device with the configs in the environment
def set_device(env, device_id=0, type='S0'):
    func = tvm.get_global_func('nnpu.set_dev', False)
    print("setting device with config file: {0}".format(env.cfg_path))
    func(int(device_id), str(type), str(env.cfg_path))

def set_dump(value):
    value = bool(value)
    func = tvm.get_global_func('nnpu.set_dump', True)
    if (func):
        func(value)

def set_profile(profiles, dir=None):
    func = tvm.get_global_func('nnpu.set_profile', False)
    flags = {'timeline': 0x1, 'memory_access_latency': 0x2}
    flag = 0
    for item in profiles:
        if (item in flags):
            flag = flag | flags[item]
    if (dir is not None):
        import os, errno
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        func(flag, dir)
    else:
        func(flag)

def get_env():
    return Environment.current

# The memory information for the compiler
@tvm.register_func("tvm.info.mem.%s" % Environment.dram_scope)
def mem_info_dram():
    spec = get_env()
    dram_cfg = spec.cfg['dram']
    return tvm.make.node("MemoryInfo",
                         unit_bits=8,
                         max_simd_bits=dram_cfg['width_per_channel'],
                         max_num_bits=dram_cfg['nchannel'] * (1 << dram_cfg['log_size_per_channel']) * 8,
                         head_address=None)

#################################
# register scratchpad memories. #
#################################
def get_scratchpad_memory_info(env, scratchpad_idx):
    from .helper import scratchpad_base_addr
    scope = env.scratchpad_scope(scratchpad_idx)
    buffer_cfg = env.get_scope_config(scope)
    if (buffer_cfg and buffer_cfg['enable']):
        return tvm.make.node("MemoryInfo",
                            unit_bits=8,
                            max_simd_bits=buffer_cfg['width_per_channel'],
                            max_num_bits=(1 << buffer_cfg['log_size']) * 8,
                            head_address=None,
                            base_address=scratchpad_base_addr[scratchpad_idx])
    else:
        raise ValueError('scratchpad buffer "{0}" is not enabled, please check config file'.format(scope))

@tvm.register_func("tvm.info.mem.{0}{1}".format(Environment.scratchpad_scope_prefix, '0'))
def mem_info_scratchpad():
    spec = get_env()
    return get_scratchpad_memory_info(spec, 0)

@tvm.register_func("tvm.info.mem.{0}{1}".format(Environment.scratchpad_scope_prefix, '1'))
def mem_info_scratchpad():
    spec = get_env()
    return get_scratchpad_memory_info(spec, 1)

@tvm.register_func("tvm.info.mem.{0}{1}".format(Environment.scratchpad_scope_prefix, '2'))
def mem_info_scratchpad():
    spec = get_env()
    return get_scratchpad_memory_info(spec, 2)

@tvm.register_func("tvm.info.mem.{0}{1}".format(Environment.scratchpad_scope_prefix, '3'))
def mem_info_scratchpad():
    spec = get_env()
    return get_scratchpad_memory_info(spec, 3)

@tvm.register_func("tvm.info.mem.{0}{1}".format(Environment.scratchpad_scope_prefix, '4'))
def mem_info_scratchpad():
    spec = get_env()
    return get_scratchpad_memory_info(spec, 4)

@tvm.register_func("tvm.info.mem.{0}{1}".format(Environment.scratchpad_scope_prefix, '5'))
def mem_info_scratchpad():
    spec = get_env()
    return get_scratchpad_memory_info(spec, 5)

@tvm.register_func("tvm.info.mem.{0}{1}".format(Environment.scratchpad_scope_prefix, '6'))
def mem_info_scratchpad():
    spec = get_env()
    return get_scratchpad_memory_info(spec, 6)

@tvm.register_func("tvm.info.mem.{0}{1}".format(Environment.scratchpad_scope_prefix, '7'))
def mem_info_scratchpad():
    spec = get_env()
    return get_scratchpad_memory_info(spec, 7)

#################################
# register accumulation buffer. #
#################################
@tvm.register_func("tvm.info.mem.%s" % Environment.acc_scope)
def mem_info_acc():
    spec = get_env()
    acc_cfg = spec.cfg['acc_buffer']
    return tvm.make.node("MemoryInfo",
                         unit_bits=8,
                         max_simd_bits=acc_cfg['width_per_channel'],
                         max_num_bits=(1 << acc_cfg['log_size']) * 8,
                         head_address=None,
                         base_address=0)


def init_default_env():
    """Iniitalize the default global env"""
    curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../../../"))
    filename = "nnpu_config.yaml"
    path_list = [
        os.path.join(curr_path, filename),
        os.path.join(proj_root, "build", filename),
        os.path.join(proj_root, filename),
        os.path.join(proj_root, "nnpu", filename),
        os.path.join(proj_root, "nnpu/config", filename)
    ]
    path_list = [p for p in path_list if os.path.exists(p)]
    if not path_list:
        raise RuntimeError(
            "Error: {} not found.make sure you have config.json in your vta root"
            .format(filename))
    return Environment(path_list[0])

# TVM related registration
@tvm.register_func("tvm.intrin.rule.default.nnpu.coproc_sync")
def coproc_sync(op):
    _ = op
    return tvm.const(0, 'int32')

@tvm.register_func("tvm.intrin.rule.default.nnpu.coproc_dep_push")
def coproc_dep_push(op):
    return make_intrin_call('void', 'DependPush',
                            op.args[0], op.args[1])

@tvm.register_func("tvm.intrin.rule.default.nnpu.coproc_dep_pop")
def coproc_dep_pop(op):
    return make_intrin_call('void', 'DependPop',
                            op.args[0], op.args[1])

Environment.current = init_default_env()
