'''
helper functions
'''
from topi import util

dtype2bytes = {'int8': 1, 'int16': 2, 'int32': 4, 'uint8': 1, 'uint16': 2, 'uint32': 4, 
               'float16': 2, 'float32': 4, 'fixed16': 2}

def dtype_bytes(dtype):
    try:
        return dtype2bytes[dtype]
    except KeyError:
        raise ValueError('unhandeled dtype: {0}'.format(dtype))

# convert scope name, also check whether scope is legal under current config
def convert_scope(env, scope_str, include_acc=False):
    scope = scope_str
    if (scope_str == 'acc'):
        scope = env.acc_scope
    elif (scope_str.startswith('buffer') or scope_str.startswith('scratchpad')):
        id = scope_str[len('buffer'):] if scope_str.startswith('buffer') \
             else scope_str[len('scratchpad'):]
        id = int(id)
        scope = env.scratchpad_scope(id)

    if (scope == env.acc_scope):
        assert include_acc, 'accumulation scope is not allowed'
        return scope
    else:
        return scope

scratchpad_base_addr = { 0: 0x0, 1: 0x10000000, 2: 0x20000000, 3: 0x30000000, 
                         4: 0x40000000, 5: 0x50000000, 6: 0x60000000, 7: 0x70000000 }

def addr_to_idx(addr):
    return (addr & 0x70000000) >> 28

def get_access_ptr(buffer, env, *args):
    '''
        used to get the address of one buffer, the actual address will be converted depends on 
        the scope.
    '''
    args = list(args)
    args.append('int32')
    addr = buffer.access_ptr(*args)
    scope = buffer.scope
    if not env.is_scratchpad_scope(scope):
        return addr
    
    cfg = env.get_scope_config(scope)
    idx = env.scratchpad_scope_to_idx(scope)
    return util.simplify(addr + scratchpad_base_addr[idx])