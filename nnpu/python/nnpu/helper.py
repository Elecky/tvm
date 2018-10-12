'''
helper functions
'''

dtype2bytes = {'int8': 1, 'int16': 2, 'int32': 4, 'uint8': 1, 'uint16': 2, 'uint32': 4, 
               'float16': 2, 'float32': 4, 'fixed16': 2}

def dtype_bytes(dtype):
    try:
        return dtype2bytes[dtype]
    except KeyError:
        raise ValueError('unhandeled dtype: {0}'.format(dtype))

# convert scope name, also check whether scope is legal under current config
def convert_scope(env, scope_str):
    scope = scope_str
    if (scope_str == 'uni'):
        scope = env.uni_scratchpad_scope
    elif (scope_str == 'vctr'):
        scope = env.vctr_scratch_scope
    elif (scope_str == 'mat'):
        scope = env.mat_scratch_scope
    design = env.cfg['scratchpad_design']
    assert not (design == 'unified') or (scope == env.uni_scratchpad_scope), \
        'illegal scope {0} in {1} scratchpad design'.format(scope_str, design)
    assert not (design == 'seperated') or \
            (scope in [env.vctr_scratch_scope, env.mat_scratch_scope]), \
            'illegal scope {0} in {1} scratchpad design'.format(scope_str, design)
    return scope