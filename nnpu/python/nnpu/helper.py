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