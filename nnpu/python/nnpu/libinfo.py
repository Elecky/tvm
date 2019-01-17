'''
'''

import ctypes
import sys
import os

def _get_lib_name():
    if sys.platform.startswith('win32'):
        return "nnpu.dll"
    if sys.platform.startswith('darwin'):
        return "libnnpu.dylib"
    return "libnnpu.so"


def find_libnnpu(optional=False):
    """Find NNPU library"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_search = [curr_path]
    lib_search += [os.path.join(curr_path, "..", "..", "..", "build",)]
    lib_search += [os.path.join(curr_path, "..", "..", "..", "build", "Release")]
    lib_name = _get_lib_name()
    lib_path = [os.path.join(x, lib_name) for x in lib_search]
    lib_found = [x for x in lib_path if os.path.exists(x)]
    if not lib_found and not optional:
        raise RuntimeError("Cannot find libvta: candidates are: " % str(lib_path))
    return lib_found

def _load_lib():
    """Load local library, assuming they are simulator."""
    lib_path = find_libnnpu(optional=False)
    if not lib_path:
        return []
    try:
        _ = ctypes.CDLL('/usr/local/lib/libyaml-cpp.so', ctypes.RTLD_GLOBAL)
    except OSError:
        raise RuntimeError('cannot load libyaml-cpp, please install yaml-cpp or \
                            modify the path here!')
    
    try:
        print(lib_path)
        return [ctypes.CDLL(lib_path[0],ctypes.RTLD_GLOBAL)]
    except OSError as err:
        print(err.strerror)
        raise RuntimeError('libnnpu not loaded')