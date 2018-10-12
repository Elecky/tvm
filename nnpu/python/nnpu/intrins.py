import tvm
from helper import dtype_bytes, convert_scope

class IntrinManager(object):

    def __init__(self, env):
        self.intrin_ctors = {}
        # the intrin cache is an dict from name to registered intrin
        self.intrin_cache = {}
        self.env = env

        # some helper dicts
        self.mode2code = {'n': 0, 'inc': 1, 'dec': 2, 'w': 3}

        # define intrin constructors here
        
        # unary vector intrin
        def vctr_unary(intrin_op, scope_in = 'uni', scope_out = 'uni', mode='w'):
            env = self.env
            cfg = self.env.cfg

            scope_in = self.get_scope(scope_in)
            scope_out = self.get_scope(scope_out)

            dtype_in, dtype_out = self.mode2dtype(mode)
            
            # the name should contain all parameters
            name = intrin_op + ';' + scope_in + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            in_shape = (cfg['vector_unit']['size'], )
            out_shape = (cfg['vector_unit']['size'], )

            op_in = tvm.placeholder(in_shape, dtype=dtype_in,
                                    name='in')
            # To Add More Intrins, just add other expression and extern function call here!!!!!
            if (intrin_op == 'VEXP'):
                if (mode == 'inc'):
                    expr = lambda i: tvm.exp(op_in[i].astype(dtype_out))
                elif (mode == 'dec'):
                    expr = lambda i: tvm.exp(op_in[i]).astype(dtype_out)
                else:
                    expr = lambda i: tvm.exp(op_in[i])
                extern_func = 'NNPU_VEXP'
            elif (intrin_op == 'VLOG'):
                if (mode == 'inc'):
                    expr = lambda i: tvm.log(op_in[i].astype(dtype_out))
                elif (mode == 'dec'):
                    expr = lambda i: tvm.log(op_in[i]).astype(dtype_out)
                else:
                    expr = lambda i: tvm.log(op_in[i])
                extern_func = 'NNPU_VLOG'
            else:
                raise ValueError('unsupported vctr unary intrin op')
            
            out = tvm.compute(out_shape, expr,
                            name = 'out')

            def lower_func(ins, outs):
                din = ins[0]
                dout = outs[0]

                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(tvm.call_extern("int32", extern_func,
                            dout.access_ptr("w", 'uint32'),
                            din.access_ptr("r", 'uint32'),
                            cfg['vector_unit']['size'],
                            self.get_mode_code(mode)
                            ))
                
                return irb.get()
            
            in_layout = self.decl_buffer(op_in, scope_in, 'in_buf')
            out_layout = self.decl_buffer(out, scope_out, 'out_buf')

            return tvm.decl_tensor_intrin(out.op, lower_func,
                                    name=name,
                                    binds={op_in: in_layout,
                                           out: out_layout})

        self.intrin_ctors['VEXP'] = vctr_unary
        self.intrin_ctors['VLOG'] = vctr_unary

        def gemm(intrin_op, shape, scope_in1 = 'uni', scope_in2 = 'uni', 
                 scope_out = 'uni', mode='inc', reduce=False):
            env = self.env
            cfg = self.env.cfg

            assert len(shape) == 3, 'shape should be tuple or list with 3 values'
            # TODO: do a shape check with cfg here!!!!
            nRowOut, factor, nColOut = shape
            assert nRowOut != 1 or nColOut != 1, 'gemm is not intended to multiply two vector!'

            scope_in1 = self.get_scope(scope_in1)
            scope_in2 = self.get_scope(scope_in2)
            scope_out = self.get_scope(scope_out)

            dtype_in, dtype_out = self.mode2dtype(mode)
            
            # the name should contain all parameters
            name = intrin_op + str(nRowOut) + '_' + str(factor) + '_' + str(nColOut) + ';' \
                   + ';' + scope_in1 + ';' + scope_in2 + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            in1 = tvm.placeholder((nRowOut, factor), dtype=dtype_in, name='in1') \
                    if nRowOut != 1 or not reduce else \
                  tvm.placeholder((factor, ), dtype=dtype_in, name='in1')
            in2 = tvm.placeholder((nColOut, factor), dtype=dtype_in, name='in2') \
                    if nColOut != 1 or not reduce else \
                  tvm.placeholder((factor, ), dtype=dtype_in, name='in2')
            k = tvm.reduce_axis((0, factor), 'k')

            # due to the limitation of tvm, we have 3 conditions to consider.
            if (nColOut == 1 and reduce):
                if (mode == 'inc'):
                    expr = lambda i: \
                        tvm.sum(in1[i, k].astype(dtype_out) * in2[k].astype(dtype_out), axis=k)
                elif (mode == 'dec'):
                    expr = lambda i: tvm.sum(in1[i, k] * in2[k], axis=k).astype(dtype_out)
                else:
                    expr = lambda i: tvm.sum(in1[i, k] * in2[k], axis=k)
                out = tvm.compute((nRowOut, ), expr, name='out')
            elif (nRowOut == 1 and reduce):
                if (mode == 'inc'):
                    expr = lambda j: \
                        tvm.sum(in1[k].astype(dtype_out) * in2[j, k].astype(dtype_out), axis=k)
                elif (mode == 'dec'):
                    expr = lambda j: tvm.sum(in1[k] * in2[j, k], axis=k).astype(dtype_out)
                else:
                    expr = lambda j: tvm.sum(in1[k] * in2[j, k], axis=k)
                out = tvm.compute((nColOut, ), expr, name='out')
            else:
                if (mode == 'inc'):
                    expr = lambda i, j: \
                        tvm.sum(in1[i, k].astype(dtype_out) * in2[j, k].astype(dtype_out), axis=k)
                elif (mode == 'dec'):
                    expr = lambda i, j: tvm.sum(in1[i, k] * in2[j, k], axis=k).astype(dtype_out)
                else:
                    expr = lambda i, j: tvm.sum(in1[i, k] * in2[j, k], axis=k)
                out = tvm.compute((nRowOut, nColOut), expr, name='out')
            in1_buf = self.decl_buffer(in1, scope_in1, 'in1')
            in2_buf = self.decl_buffer(in2, scope_in2, 'in2')
            out_buf = self.decl_buffer(out, scope_out, 'out')

            def lower_func(ins, outs):
                din1, din2 = ins[0], ins[1]
                dout = outs[0]

                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(tvm.call_extern("int32", 'NNPU_Gemm',
                            nRowOut, factor, nColOut,
                            dout.access_ptr('w', 'uint32'),
                            din1.access_ptr('r', 'uint32'),
                            din2.access_ptr('r', 'uint32'),
                            self.get_mode_code(mode)
                            ))
                
                return irb.get()

            return tvm.decl_tensor_intrin(out.op, lower_func,
                                          name=name,
                                          binds={in1: in1_buf,
                                                 in2: in2_buf,
                                                 out: out_buf})
        self.intrin_ctors['GEMM'] = gemm

        def vctr_binary(intrin_op, scope_in1 = 'uni', scope_in2 = 'uni', 
                 scope_out = 'uni', mode='n'):
            env = self.env
            cfg = self.env.cfg

            scope_in1 = self.get_scope(scope_in1)
            scope_in2 = self.get_scope(scope_in2)
            scope_out = self.get_scope(scope_out)

            dtype_in, dtype_out = self.mode2dtype(mode)

            # the name should contain all parameters
            name = intrin_op + scope_in1 + ';' + scope_in2 + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            shape = (cfg['vector_unit']['size'], )

            in1 = tvm.placeholder(shape, dtype_in, 'in1')
            in2 = tvm.placeholder(shape, dtype_in, 'in2')

            def expr_template(x, y, func):
                if (mode == 'inc'):
                    return func(x.astype(dtype_out), y.astype(dtype_out))
                elif (mode == 'dec'):
                    return func(x, y).astype(dtype_out)
                else:
                    return func(x, y)

            if (intrin_op == 'VAddV'):
                expr = lambda i: expr_template(in1[i], in2[i], lambda x, y: x + y)
                extern_func = 'NNPU_VAddV'
            elif (intrin_op == 'VSubV'):
                expr = lambda i: expr_template(in1[i], in2[i], lambda x, y: x - y)
                extern_func = 'NNPU_VSubV'
            elif (intrin_op == 'VMulV'):
                expr = lambda i: expr_template(in1[i], in2[i], lambda x, y: x * y)
                extern_func = 'NNPU_VMulV'
            elif (intrin_op == 'VDivV'):
                expr = lambda i: expr_template(in1[i], in2[i], lambda x, y: x / y)
                extern_func = 'NNPU_VDivV'
            elif (intrin_op == 'VGTMV'):
                expr = lambda i: expr_template(in1[i], in2[i], 
                                    lambda x, y: tvm.select(x > y, x, y))
                extern_func = 'NNPU_VGTMV'
            else:
                raise ValueError('unhandled intrin_op in vctr_binary')

            out = tvm.compute(shape, expr, 'out')
            in1_buf = self.decl_buffer(in1, scope_in1, 'in1_buf')
            in2_buf = self.decl_buffer(in2, scope_in2, 'in2_buf')
            out_buf = self.decl_buffer(out, scope_out, 'out_buf')
            
            def lower_func(ins, outs):
                din1, din2 = ins[0], ins[1]
                dout = outs[0]

                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(tvm.call_extern("int32", extern_func,
                            dout.access_ptr('w', 'uint32'),
                            din1.access_ptr('r', 'uint32'),
                            din2.access_ptr('r', 'uint32'),
                            shape[0],
                            self.get_mode_code(mode)
                            ))
                
                return irb.get()

            return tvm.decl_tensor_intrin(out.op, lower_func,
                                          name=name,
                                          binds={in1: in1_buf,
                                                 in2: in2_buf,
                                                 out: out_buf})
        self.intrin_ctors['VAddV'] = vctr_binary
        self.intrin_ctors['VSubV'] = vctr_binary
        self.intrin_ctors['VMulV'] = vctr_binary
        self.intrin_ctors['VDivV'] = vctr_binary
        self.intrin_ctors['VGTMV'] = vctr_binary

    def get(self, intrin_op, **kwargs):
        assert intrin_op in self.intrin_ctors, 'can not find constructor for intrin {0}'.\
            format(intrin_op)
        if ('mode' in kwargs):
            assert kwargs['mode'] in self.mode2code, \
                'illegal mode value {0}'.format(kwargs['mode'])
        return self.intrin_ctors[intrin_op](intrin_op, **kwargs)

    # convert scope name, also check whether scope is legal under current config
    def get_scope(self, scope_str):
        return convert_scope(self.env, scope_str)
    
    def decl_buffer(self, tensor, scope, buf_name):
        dtype_bits = dtype_bytes(tensor.dtype) * 8
        return tvm.decl_buffer(
                tensor.shape, tensor.dtype, buf_name, scope=scope,
                data_alignment=self.env.scope2config(scope)['width_per_channel'] / 8,
                offset_factor=self.env.scope2config(scope)['width_per_channel'] / dtype_bits)

    def mode2dtype(self, mode):
        cfg = self.env.cfg
        assert mode in ['w', 'n', 'inc', 'dec'], 'invalid mode string'

        dtype_in = cfg['dtype_w'] if mode in ['w', 'dec'] else cfg['dtype_n']
        dtype_out = cfg['dtype_w'] if mode in ['w', 'inc'] else cfg['dtype_n']
        return dtype_in, dtype_out

    def get_mode_code(self, mode_str):
        return self.mode2code[mode_str]
