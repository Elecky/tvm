'''the environment and context of NNPU hardware simulator
some code are from tvm/vta
'''

import os
import json
import copy
import tvm
import yaml

from intrins import declare_intrins
from intrins import IntrinManager

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
    uni_scratchpad_scope = 'local.nnpu_scratchpad'
    vctr_scratch_scope = 'local.nnpu_vscratchpad'
    mat_scratch_scope = 'local.nnpu_mscratchpad'
    # compiler pragmas
    dma_copy_pragma = 'nnpu_dma_copy'
    scratchpad_ls = 'nnpu_scratchpad_ls'

    def __init__(self, cfg_path):
        self.cfg = {}

        cfg = yaml.load(open(cfg_path))
        self.cfg_path = cfg_path
        self.cfg.update(cfg)

        self.nnpu_axis = tvm.thread_axis('nnpu')

        self.intrins = IntrinManager(self)
        pass

    def __enter__(self):
        self.last_env = Environment.current
        Environment.current = self
        set_device(self)
        return self

    def __exit__(self, ptype, value, trace):
        Environment.current = self.last_env
        # reset device based on the last Environment
        set_device(Environment.current)

    def scope2config(self, scope):
        key = None
        if (scope == self.uni_scratchpad_scope):
            key = 'scratchpad'
        elif (scope == self.vctr_scratch_scope):
            key = 'vctr_scratchpad'
        elif (scope == self.mat_scratch_scope):
            key = 'mat_scratchpad'
        else:
            raise ValueError('illegal scope name')
        return self.cfg[key]

# set device with the configs in the environment
def set_device(env):
    func = tvm.get_global_func('nnpu.set_dev',True)
    func('S0', env.cfg_path)
    pass

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

@tvm.register_func("tvm.info.mem.{0}".format(Environment.uni_scratchpad_scope))
def mem_info_scratchpad():
    spec = get_env()
    if (spec.cfg['scratchpad_design'] == 'unified'):
        buffer_cfg = spec.cfg['scratchpad']
        return tvm.make.node("MemoryInfo",
                             unit_bits=8,
                             max_simd_bits=buffer_cfg['width_per_channel'],
                             max_num_bits=buffer_cfg['nchannel'] * (1 << buffer_cfg['log_size_per_channel']) * 8,
                             head_address=None)
    else:
        return None

@tvm.register_func("tvm.info.mem.{0}".format(Environment.vctr_scratch_scope))
def mem_info_vscratchpad():
    raise NotImplementedError

@tvm.register_func("tvm.info.mem.{0}".format(Environment.mat_scratch_scope))
def mem_info_mscratchpad():
    raise NotImplementedError

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
    return tvm.call_extern(
        "int32", "NNPUSynchronize", 1<<31)

def init():
    Environment.current = init_default_env()
    # set initial device
    set_device(Environment.current)