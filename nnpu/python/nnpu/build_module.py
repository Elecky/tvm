''' 
wrapper of tvm.lower and tvm.build for nnpu 
some code are from tvm.vta
'''

import tvm
from . import ir_pass
from .environment import get_env

def early_rewrite(stmt):
    """Try to do storage rewrite in early pass."""
    try:
        return tvm.ir_pass.StorageRewrite(stmt)
    except tvm.TVMError:
        return stmt

def build_config(debug_flag=0, **kwargs):
    """Build a build config for VTA.

    Parameters
    ----------
    debug_flag : int
        The dbeug flag to be passed.

    kwargs : dict
        Additional configurations.

    Returns
    -------
    build_config: BuildConfig
        The build config that can be used in TVM.

    Example
    --------
    .. code-block:: python

      # build a vta module.
      with vta.build_config():
          vta_module = tvm.build(s, ...)
    """
    # FIXME: the tvm StorageRewrite may cause some wired problems, check it back!!@!!!!
    env = get_env()
    pass_list = [(1, early_rewrite),
                 (1, ir_pass.inject_dma_intrin),
                 (1, ir_pass.inject_scratchpad_ls),
                 (1, ir_pass.inject_scratchpad_copy),
                 (1, ir_pass.inject_accTobuffer),
                 (1, tvm.ir_pass.CoProcSync),
                ]

    return tvm.build_config(add_lower_pass=pass_list, **kwargs)


def lower(*args, **kwargs):
    """Thin wrapper of tvm.lower

    This wrapper automatically applies VTA's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.lower : The original TVM's lower function
    """
    cfg = tvm.build_module.current_build_config()
    if not cfg.add_lower_pass:
        with build_config():
            return tvm.lower(*args, **kwargs)
    return tvm.lower(*args, **kwargs)


def build(*args, **kwargs):
    """Thin wrapper of tvm.build

    This wrapper automatically applies VTA's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.build : The original TVM's build function
    """
    cfg = tvm.build_module.current_build_config()
    if not cfg.add_lower_pass:
        with build_config():
            return tvm.build(*args, **kwargs)
    return tvm.build(*args, **kwargs)
