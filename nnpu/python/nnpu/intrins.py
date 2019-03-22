import struct
import tvm
from helper import dtype_bytes, convert_scope

def make_intrin_call(dtype, name, *args, **kwargs):
    """ Build a llvm.NNPU intrinsic function call who has side-effect.
    Parameters
    ----------
    dtype : str
        The data type of the result. can be void to indicate no return value.
    name : str
        The name of the llvm intrinsic function 'without' llvm.NNPU prefix.
    
    num_signature : int
        I don't sure what this is, maybe used with overloaded llvm intrinsic
            function matching.
    args : list
        Poistional arguments.
    """
    name = 'llvm.NNPU.' + name
    if ('num_signature' in kwargs):
        num_signature = kwargs['num_signature']
    else:
        num_signature = 0
    return tvm.call_llvm_intrin_with_side_effect(
                    dtype, name, tvm.const(num_signature, 'uint32'), *args
                )

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
            if (intrin_op == 'VExp'):
                if (mode == 'inc'):
                    expr = lambda i: tvm.exp(op_in[i].astype(dtype_out))
                elif (mode == 'dec'):
                    expr = lambda i: tvm.exp(op_in[i]).astype(dtype_out)
                else:
                    expr = lambda i: tvm.exp(op_in[i])
                intrin_func = 'VExp'
            elif (intrin_op == 'VLog'):
                if (mode == 'inc'):
                    expr = lambda i: tvm.log(op_in[i].astype(dtype_out))
                elif (mode == 'dec'):
                    expr = lambda i: tvm.log(op_in[i]).astype(dtype_out)
                else:
                    expr = lambda i: tvm.log(op_in[i])
                intrin_func = 'VLog'
            else:
                raise ValueError('unsupported vctr unary intrin op')
            
            out = tvm.compute(out_shape, expr,
                            name = 'out')

            def lower_func(ins, outs):
                din = ins[0]
                dout = outs[0]

                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(make_intrin_call("void", intrin_func,
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

        self.intrin_ctors['VExp'] = vctr_unary
        self.intrin_ctors['VLog'] = vctr_unary

        def vctr_imm(intrin_op, scope_in = 'uni', scope_out = 'uni', imm_value = 1 , mode = 'w'):
            env = self.env
            cfg = self.env.cfg
            scope_in = self.get_scope(scope_in)
            scope_out = self.get_scope(scope_out)
            dtype_in, dtype_out = self.mode2dtype(mode)
            
            if (isinstance(imm_value, type(tvm.const(0)))):
                imm = imm_value.astype(dtype_in)
            else:
                imm = tvm.const(imm_value, dtype_in)
            name = intrin_op + ';' + scope_in + ';' + scope_out + ';' +str(imm_value)+';'+ mode
            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            in_shape = (cfg['vector_unit']['size'], )
            out_shape = (cfg['vector_unit']['size'], )
            op_in = tvm.placeholder(in_shape, dtype=dtype_in,
                                    name='in')

            def expr_template(in1, imm, func):
                if (mode == 'inc'):
                    return lambda i: func(in1[i].astype(dtype_out), imm.astype(dtype_out))
                elif (mode == 'dec'):
                    return lambda i: func(in1[i], imm).astype(dtype_out)
                else:
                    return lambda i: func(in1[i], imm)

            if (intrin_op == 'VAddI'):
                expr = expr_template(op_in, imm, lambda x, y: x + y)
                intrin_func = 'VAddI'
            elif (intrin_op == 'VSubI'):
                expr = expr_template(op_in, imm, lambda x, y: x - y)
                intrin_func = 'VSubI'
            elif (intrin_op == 'VMulI'):
                expr = expr_template(op_in, imm, lambda x, y: x * y)
                intrin_func = 'VMulI'
            elif (intrin_op == 'VDivI'):
                expr = expr_template(op_in, imm, lambda x, y: x / y)
                intrin_func = 'VDivI'
            elif (intrin_op == 'VGTMI'):
                expr = expr_template(op_in, imm, lambda x, y: tvm.max(x, y))
                intrin_func = 'VGTMI'
            elif (intrin_op == 'ISubV'):
                expr = expr_template(op_in,imm, lambda x, y: y - x)
                intrin_func = 'ISubV'
            elif (intrin_op == 'IDivV'):
                expr = expr_template(op_in,imm,lambda x , y : y / x)
                intrin_func = 'IDivV'
            else:
                raise ValueError('unsupported vctr Imm intrin op')
            out = tvm.compute(out_shape, expr,
                            name = 'out')
            def lower_func(ins, outs):
                din = ins[0]
                dout = outs[0]

                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(make_intrin_call("void", intrin_func,
                            dout.access_ptr("w", 'uint32'),
                            din.access_ptr("r", 'uint32'),
                            tvm.const(imm_value, 'float64'),
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

        self.intrin_ctors['VAddI'] = vctr_imm
        self.intrin_ctors['VSubI'] = vctr_imm
        self.intrin_ctors['VMulI'] = vctr_imm
        self.intrin_ctors['VDivI'] = vctr_imm
        self.intrin_ctors['VGTMI'] = vctr_imm
        self.intrin_ctors['ISubV'] = vctr_imm
        self.intrin_ctors['IDivV'] = vctr_imm

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
            scope_out = self.get_scope(scope_out, include_acc=True)

            dtype_in, dtype_out = self.mode2dtype(mode)
            
            # the name should contain all parameters
            name = intrin_op + str(nRowOut) + '_' + str(factor) + '_' + str(nColOut) + ';' \
                   + ';' + scope_in1 + ';' + scope_in2 + ';' + scope_out + ';' + mode + \
                   ';' + str(reduce)

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
            in1_strides, in2_strides, out_strides = None, None, None
            if (nColOut == 1 and reduce):
                if (mode == 'inc'):
                    expr = lambda i: \
                        tvm.sum(in1[i, k].astype(dtype_out) * in2[k].astype(dtype_out), axis=k)
                elif (mode == 'dec'):
                    expr = lambda i: tvm.sum(in1[i, k] * in2[k], axis=k).astype(dtype_out)
                else:
                    expr = lambda i: tvm.sum(in1[i, k] * in2[k], axis=k)
                out = tvm.compute((nRowOut, ), expr, name='out')
                in1_strides = [tvm.var('s1'), 1]
            elif (nRowOut == 1 and reduce):
                if (mode == 'inc'):
                    expr = lambda j: \
                        tvm.sum(in1[k].astype(dtype_out) * in2[j, k].astype(dtype_out), axis=k)
                elif (mode == 'dec'):
                    expr = lambda j: tvm.sum(in1[k] * in2[j, k], axis=k).astype(dtype_out)
                else:
                    expr = lambda j: tvm.sum(in1[k] * in2[j, k], axis=k)
                out = tvm.compute((nColOut, ), expr, name='out')
                in2_strides = [tvm.var('s2'), 1]
            else:
                if (mode == 'inc'):
                    expr = lambda i, j: \
                        tvm.sum(in1[i, k].astype(dtype_out) * in2[j, k].astype(dtype_out), axis=k)
                elif (mode == 'dec'):
                    expr = lambda i, j: tvm.sum(in1[i, k] * in2[j, k], axis=k).astype(dtype_out)
                else:
                    expr = lambda i, j: tvm.sum(in1[i, k] * in2[j, k], axis=k)
                out = tvm.compute((nRowOut, nColOut), expr, name='out')
                in1_strides = [tvm.var('s1'), 1]
                in2_strides = [tvm.var('s2'), 1]
                out_strides = [tvm.var('s3'), 1]
            
            in1_buf = self.decl_buffer(in1, scope_in1, 'in1', strides=in1_strides)
            in2_buf = self.decl_buffer(in2, scope_in2, 'in2', strides=in2_strides)
            out_buf = self.decl_buffer(out, scope_out, 'out', strides=out_strides)

            def lower_func(ins, outs):
                ins = self.get_ins(ins, 'in1', 'in2')
                din1, din2 = ins[0], ins[1]
                dout = outs[0]

                in1_row_stride = din1.strides[0] * dtype_bytes(dtype_in) \
                                 if din1.strides else 0
                in2_row_stride = din2.strides[0] * dtype_bytes(dtype_in) \
                                 if din2.strides else 0
                if (nColOut == 1 and reduce):
                    out_row_stride = 1
                elif (nRowOut == 1 and reduce):
                    out_row_stride = 0
                else:
                    out_row_stride = dout.strides[0]
                out_row_stride = out_row_stride * dtype_bytes(dtype_out)

                init = self.emit_acc_init(dout.access_ptr('w', 'uint32'),
                                nRowOut, nColOut, out_row_stride, mode, 0)

                def calc(toAccBuf, doAcc):
                    irb = tvm.ir_builder.create()
                    irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                    ptr_type = 'rw' if doAcc else 'w'
                    irb.emit(make_intrin_call("void", 'GEMM',
                                nRowOut, factor, nColOut,
                                dout.access_ptr(ptr_type, 'uint32'),
                                out_row_stride,
                                din1.access_ptr('r', 'uint32'),
                                in1_row_stride,
                                din2.access_ptr('r', 'uint32'),
                                in2_row_stride,
                                self.get_mode_code(mode),
                                toAccBuf, doAcc
                                ))
                    return irb.get()
                
                if (scope_out == env.acc_scope):
                    return calc(True, False), init, calc(True, True)
                else:
                    return calc(False, False)

            return tvm.decl_tensor_intrin(out.op, lower_func,
                                          name=name,
                                          binds={in1: in1_buf,
                                                 in2: in2_buf,
                                                 out: out_buf})
        self.intrin_ctors['GEMM'] = gemm

        def mat_imm(intrin_op, shape, imm_value, scope_in = 'uni',
                 scope_out = 'uni', mode='inc',reduce=False):
            env = self.env
            cfg = self.env.cfg
            
            assert len(shape) == 2, 'shape should be tuple or list with 2 values'
            # TODO: do a shape check with cfg here!!!!
            nRow, nCol  = shape
            
            assert nRow != 1 or nCol != 1, 'gemm is not intended to multiply two vector!'

            scope_in = self.get_scope(scope_in)
            scope_out = self.get_scope(scope_out)

            dtype_in, dtype_out = self.mode2dtype(mode)

            imm_value = float(imm_value)  # convert 
            imm = tvm.const(imm_value, dtype_in)
            # the name should contain all parameters
            name = intrin_op + str(nRow) + '_'+ str(nCol) +  '_'+ str(imm.value) +  ';' \
                   + ';' + scope_in + ';' + scope_out + ';' + mode + \
                   ';' + str(reduce)

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            in1 = tvm.placeholder((nRow, nCol), dtype=dtype_in, name='in1')
            def expr_template(in1, imm, func):
                if (mode == 'inc'):
                    return lambda i , j: func(in1[i][j].astype(dtype_out), imm.astype(dtype_out))
                elif (mode == 'dec'):
                    return lambda i , j: func(in1[i][j], imm).astype(dtype_out)
                else:
                    return lambda i , j: func(in1[i][j], imm)
            # due to the limitation of tvm, we have 3 conditions to consider.
            if (intrin_op == 'MAddI'):
                expr = expr_template(in1, imm, lambda x, y: x + y)
                intrin_func = 'MAddI'
            elif (intrin_op == 'MMulI'):
                expr = expr_template(in1, imm, lambda x, y: x * y)
                intrin_func = 'MMulI'
            elif (intrin_op == 'ISubM'):
                expr = expr_template(in1, imm, lambda x, y: y - x )
                intrin_func = 'ISubM'
            out = tvm.compute((nRow, nCol), expr, name='out')
            in1_buf = self.decl_buffer(in1, scope_in, 'in')
            out_buf = self.decl_buffer(out, scope_out, 'out')
            def lower_func(ins, outs):
                din = ins[0]
                dout = outs[0]
                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(make_intrin_call("void", intrin_func,
                            dout.access_ptr('w', 'uint32'),
                            din.access_ptr('r', 'uint32'),
                            tvm.const(imm_value, 'float64'), 
                            nRow, nCol, 
                            self.get_mode_code(mode)
                            ))
                return irb.get()
            return tvm.decl_tensor_intrin(out.op, lower_func,
                                          name=name,
                                          binds={in1: in1_buf,
                                                 out: out_buf})
        self.intrin_ctors['MAddI'] = mat_imm
        self.intrin_ctors['MMulI'] = mat_imm
        self.intrin_ctors['ISubM'] = mat_imm
        
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
                    return lambda i: func(x[i].astype(dtype_out), y[i].astype(dtype_out))
                elif (mode == 'dec'):
                    return lambda i: func(x[i], y[i]).astype(dtype_out)
                else:
                    return lambda i: func(x[i], y[i])

            if (intrin_op == 'VAddV'):
                expr = expr_template(in1, in2, lambda x, y: x + y)
                intrin_func = 'VAddV'
            elif (intrin_op == 'VSubV'):
                expr = expr_template(in1, in2, lambda x, y: x - y)
                intrin_func = 'VSubV'
            elif (intrin_op == 'VMulV'):
                expr = expr_template(in1, in2, lambda x, y: x * y)
                intrin_func = 'VMulV'
            elif (intrin_op == 'VDivV'):
                expr = expr_template(in1, in2, lambda x, y: x / y)
                intrin_func = 'VDivV'
            elif (intrin_op == 'VGTMV'):
                expr = expr_template(in1, in2, 
                                    lambda x, y: tvm.max(x, y))
                intrin_func = 'VGTMV'
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
                irb.emit(make_intrin_call("void", intrin_func,
                            dout.access_ptr('w', 'int32'),
                            din1.access_ptr('r', 'int32'),
                            din2.access_ptr('r', 'int32'),
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

        def vctr_merge(intrin_op, scope_in = 'uni', scope_out = 'uni', mode='n', nDim=2):
            env = self.env
            cfg = self.env.cfg

            assert mode in ['n', 'w'], 'merge intrin can only have mode n or w'
            assert nDim >= 2, 'merge intrin requires nDim >= 2'
            scope_in = self.get_scope(scope_in)
            scope_out = self.get_scope(scope_out)

            dtype_in, dtype_out = self.mode2dtype(mode)

            # the name should contain all parameters
            name = intrin_op + scope_in + ';' + scope_out + ';' + str(nDim) + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            shape_in = [1] * (nDim - 1)
            shape_in.append(cfg['vector_unit']['size'])
            #shape_in = tuple(shape_in)
            shape_out = [1] * (nDim - 2)
            shape_out.append(cfg['vector_unit']['size'])
            #shape_out = (cfg['vector_unit']['size'], )

            in1 = tvm.placeholder(shape_in, dtype_in, 'in1')
            k = tvm.reduce_axis((0, 1), 'k_d')
            num = 0.0
            if (intrin_op == 'VAddMerge'):
                expr = lambda *i: tvm.sum(in1(k, *i), axis=k)
                num = 0.0
                intrin_func = 'VAddV'
            elif(intrin_op == 'VMulMerge'):
                expr = lambda *i: tvm.sum(in1(k, *i), axis=k)
                num = 1.0
                intrin_func = 'VMulV'
            elif(intrin_op == 'VGTMMerge'):
                expr = lambda *i: tvm.sum(in1(k, *i), axis=k)
                num = float('-inf')
                intrin_func = 'VGTMV'
            else:
                raise ValueError('unsupported op in vctr_merge: ' + intrin_op)
            
            out = tvm.compute(shape_out, expr, 'out')

            # create strides array
            strides = []
            for i in range(nDim - 1):
                strides.append(tvm.var('s{0}'.format(i)))
            strides.append(1)
            in_buf = self.decl_buffer(in1, scope_in, 'in_buf', strides=strides)
            out_buf = self.decl_buffer(out, scope_out, 'out_buf')

            def lower_func(ins, outs):
                din = ins[0]
                dout = outs[0]

                init = self.emit_memset(dout.access_ptr('w', 'uint32'), shape_out[-1], 
                            dtype_bytes(dtype_out), num , mode)

                def comp():
                    irb = tvm.ir_builder.create()
                    irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                    irb.emit(make_intrin_call("void", intrin_func,
                            dout.access_ptr('w', 'uint32'),
                            din.access_ptr('r', 'uint32'),
                            dout.access_ptr('r', 'uint32'),
                            shape_out[-1],
                            self.get_mode_code(mode)
                            ))
                
                    return irb.get()
                return None, init, comp()
            
            return tvm.decl_tensor_intrin(out.op, lower_func, name=name, 
                                          binds={in1: in_buf,
                                                 out: out_buf})
        self.intrin_ctors['VAddMerge'] = vctr_merge
        self.intrin_ctors['VMulMerge'] = vctr_merge
        self.intrin_ctors['VGTMMerge'] = vctr_merge


        def vctr_dot_product(intrin_op, scope_in1 = 'uni', scope_in2 = 'uni', scope_out = 'uni',
                             mode='n'):
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
            # decalre tensors
            shape = (cfg['vector_unit']['size'], )
            in1 = tvm.placeholder(shape, dtype_in, 'in1')
            in2 = tvm.placeholder(shape, dtype_out, 'in1')

            k = tvm.reduce_axis((0, shape[0]), 'k')
            if (mode == 'inc'):
                expr = lambda i: tvm.sum(in1[k].astype(dtype_out) * in2[k].astype(dtype_out), axis=k)
            elif (mode == 'dec'):
                expr = lambda i: tvm.sum(in1[k] * in2[k], axis=k).astype(dtype_out)
            else:
                expr = lambda i: tvm.sum(in1[k] * in2[k], axis=k)
            out = tvm.compute((1, ), expr, 'out')
            # declare buffers
            in1_buf = self.decl_buffer(in1, scope_in1, 'in1_buf')
            in2_buf = self.decl_buffer(in2, scope_in1, 'in1_buf')
            # the output buffer is different from normal buffer
            out_buf = tvm.decl_buffer(
                out.shape, out.dtype, 'out_buf', scope=scope_out,
                data_alignment=dtype_bytes(out.dtype), offset_factor=1)
            
            def lower_func(ins, outs):
                din1, din2 = ins[0], ins[1]
                dout = outs[0]

                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(make_intrin_call("void", 'VDotV',
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
        
        self.intrin_ctors['VDotV'] = vctr_dot_product

        def vctr_reduce(intrin_op, scope_in='uni', scope_out='uni', mode='inc'):
            env = self.env
            cfg = self.env.cfg

            scope_in = self.get_scope(scope_in)
            scope_out = self.get_scope(scope_out)

            dtype_in, dtype_out = self.mode2dtype(mode)
            
            # the name should contain all parameters
            name = intrin_op + ';' + scope_in + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            shape = (cfg['vector_unit']['size'], )

            op_in = tvm.placeholder(shape, dtype=dtype_in, name='in')
            
            def expr_template(x, func, k):
                if (mode == 'inc'):
                    return lambda i: func(x.astype(dtype_out)[k], k)
                elif (mode == 'dec'):
                    return lambda i: func(x[k], k).astype(dtype_out)
                else:
                    return lambda i: func(x[k], k)

            k = tvm.reduce_axis((0, shape[0]), 'k')
            if (intrin_op == 'VReduceSum'):
                expr = expr_template(op_in, tvm.sum, k)
                intrin_func = 'VReduceSum'
            elif (intrin_op == 'VReduceMax'):
                expr = expr_template(op_in, tvm.max, k)
                intrin_func = 'VReduceMax'
            elif (intrin_op == 'VReduceMin'):
                expr = expr_template(op_in, tvm.min, k)
                intrin_func = 'VReduceMin'
            else:
                raise ValueError("unimplemented vctr reduce op")
            out = tvm.compute((1,), expr, 'out')

            in_buf = self.decl_buffer(op_in, scope_in, 'in_buf')
            out_buf = tvm.decl_buffer(
                out.shape, out.dtype, 'out_buf', scope=scope_out,
                data_alignment=dtype_bytes(out.dtype), offset_factor=1)

            def lower_func(ins, outs):
                din1 = ins[0]
                dout = outs[0]

                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(make_intrin_call("void", intrin_func,
                            dout.access_ptr('w', 'uint32'),
                            din1.access_ptr('r', 'uint32'),
                            shape[0],
                            self.get_mode_code(mode)
                            ))
                
                return irb.get()

            return tvm.decl_tensor_intrin(out.op, lower_func,
                                          name=name,
                                          binds={op_in: in_buf,
                                                 out: out_buf})
        
        self.intrin_ctors['VReduceSum'] = vctr_reduce
        self.intrin_ctors['VReduceMax'] = vctr_reduce
        self.intrin_ctors['VReduceMin'] = vctr_reduce

        def vctr_reduce_key(intrin_op, scope_in1='uni', scope_out1='uni',scope_out2='uni', mode='inc'):
            env = self.env
            cfg = self.env.cfg

            scope_in1 = self.get_scope(scope_in1)
            scope_out1 = self.get_scope(scope_out1)
            scope_out2 = self.get_scope(scope_out2)
            dtype_in, dtype_out = self.mode2dtype(mode)
            
            # the name should contain all parameters
            name = intrin_op + ';' + scope_in1 +  ';' + scope_out1 + ';' +scope_out2 + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            shape = (5,cfg['vector_unit']['size'])

            op_in1 = tvm.placeholder(shape, dtype=dtype_in, name='in1')
            k = tvm.reduce_axis((0, shape[1]), 'k')
            extern_func = 'NNPU_VReduceKey'
            def fcombine(x, y):
                lhs = tvm.select((x[1] >= y[1]), x[0], y[0])
                rhs = tvm.select((x[1] >= y[1]), x[0], y[0])
                return lhs,rhs

            def fidentity(t0, t1):
                return tvm.const(-1, t0), tvm.min_value(t1)

            argmax = tvm.comm_reducer(fcombine, fidentity, name='argmax')
            out1 , out2 = tvm.compute((5,), lambda i: argmax((op_in1[i,k], op_in1[4,k]), axis=k), 'out')

            in1_buf = self.decl_buffer(op_in1, scope_in1, 'in1_buf')
            out1_buf = tvm.decl_buffer(
                out1.shape, out1.dtype, 'out1_buf', scope=scope_out1,
                data_alignment=dtype_bytes(out1.dtype), offset_factor=1)
            out2_buf = tvm.decl_buffer(
                out2.shape, out2.dtype, 'out2_buf', scope=scope_out2,
                data_alignment=dtype_bytes(out2.dtype), offset_factor=1)

            
            def lower_func(ins, outs):
                din1 = ins[0]
                dout1 = outs[0]
                dout2 = outs[1]
                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(tvm.call_extern("int32", extern_func,
                            dout1.access_ptr('w', 'uint32'),
                            dout2.access_ptr('w', 'uint32'),
                            din1.access_ptr('r', 'uint32'),
                            shape[0]*shape[1],
                            self.get_mode_code(mode)
                            ))
                
                return irb.get()

            return tvm.decl_tensor_intrin(out1.op, lower_func,
                                          name=name,
                                          binds={op_in1: in1_buf,
                                                 out1: out1_buf,
                                                 out2: out2_buf})
        
        self.intrin_ctors['VReduceKey'] = vctr_reduce_key

        def mat_binary(intrin_op, shape, scope_in1='uni', scope_in2='uni', scope_out='uni',
                       mode='n'):
            env = self.env
            cfg = self.env.cfg

            scope_in1 = self.get_scope(scope_in1)
            scope_in2 = self.get_scope(scope_in2)
            scope_out = self.get_scope(scope_out)

            dtype_in, dtype_out = self.mode2dtype(mode)

            # TODO: validate shape with cfg
            assert len(shape) == 2, 'the length of shape should be 2'
            nRow, nCol = shape
            
            # the name should contain all parameters
            name = intrin_op + ';' + str(nRow) + '_' + str(nCol) + '_' \
                    + scope_in1 + ';' + scope_in2 + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]
            
            in1 = tvm.placeholder(shape, dtype_in, 'in1')
            in2 = tvm.placeholder(shape, dtype_in, 'in2')

            def expr_template(x, y, func):
                if (mode == 'inc'):
                    return lambda *i: func(x(*i).astype(dtype_out), y(*i).astype(dtype_out))
                elif (mode == 'dec'):
                    return lambda *i: func(x(*i), y(*i)).astype(dtype_out)
                else:
                    return lambda *i: func(x(*i), y(*i))

            if (intrin_op == 'MAddM'):
                expr = expr_template(in1, in2, lambda x, y: x + y)
                intrin_func = 'MAddM'
            elif (intrin_op == 'MSubM'):
                expr = expr_template(in1, in2, lambda x, y: x - y)
                intrin_func = 'MSubM'
            elif (intrin_op == 'MMulM'):
                expr = expr_template(in1, in2, lambda x, y: x * y)
                intrin_func = 'MMulM'
            else:
                raise ValueError('unsupported mat binary op')
            out = tvm.compute(shape, expr, 'out')

            in1_buf = self.decl_buffer(in1, scope_in1, 'in1_buf', strides=(tvm.var('s1'), 1))
            in2_buf = self.decl_buffer(in2, scope_in2, 'in2_buf', strides=(tvm.var('s2'), 1))
            out_buf = self.decl_buffer(out, scope_out, 'out_buf', strides=(tvm.var('s3'), 1))
            
            def lower_func(ins, outs):
                din1, din2 = ins[0], ins[1]
                dout = outs[0]

                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(make_intrin_call("void", intrin_func,
                            dout.access_ptr('w', 'uint32'), 
                            dout.strides[0] * dtype_bytes(dtype_out),
                            din1.access_ptr('r', 'uint32'), 
                            din1.strides[0] * dtype_bytes(dtype_in),
                            din2.access_ptr('r', 'uint32'), 
                            din2.strides[0] * dtype_bytes(dtype_in),
                            shape[0], shape[1],
                            self.get_mode_code(mode)
                            ))
                
                return irb.get()

            return tvm.decl_tensor_intrin(out.op, lower_func,
                                          name=name,
                                          binds={in1: in1_buf,
                                                 in2: in2_buf,
                                                 out: out_buf})
        self.intrin_ctors['MAddM'] = mat_binary
        self.intrin_ctors['MSubM'] = mat_binary
        self.intrin_ctors['MMulM'] = mat_binary

        def mat_merge(intrin_op, shape ,scope_in = 'uni', scope_out = 'uni', mode='n'):
            env = self.env
            cfg = self.env.cfg

            assert mode in ['n', 'w'], 'merge intrin can only have mode n or w'

            scope_in = self.get_scope(scope_in)
            scope_out = self.get_scope(scope_out)
            ndim,nrow,ncol=shape
            dtype_in, dtype_out = self.mode2dtype(mode)
            # print('THIS IS SHAP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # print(shape)
            # print(nrow*ncol)
            assert len(shape) == 3, 'the length of shape should be 3'
            # the name should contain all parameters
            name = intrin_op + scope_in + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            shape_in = (1,nrow,ncol)
            shape_out = (nrow,ncol)

            in1 = tvm.placeholder(shape_in, dtype_in, 'in1')
            k = tvm.reduce_axis((0, 1), 'k_d')
            num = 0.0
            if (intrin_op == 'MAddMerge'):
                expr = lambda i,j: tvm.sum(in1[k, i,j], axis=k)
                num = 0.0
                intrin_func = 'MAddM'
            elif(intrin_op == 'MMulMerge'):
                expr = lambda i,j: tvm.sum(in1[k, i,j], axis=k)
                num = 1.0
                intrin_func = 'MMulM'
            else:
                raise ValueError('unsupported op in mat_merge: ' + intrin_op)
            
            out = tvm.compute(shape_out, expr, 'out')

            in_buf = self.decl_buffer(in1, scope_in, 'in_buf')
            out_buf = self.decl_buffer(out, scope_out, 'out_buf')

            def lower_func(ins, outs):
                din = ins[0]
                dout = outs[0]

                init = self.emit_memset(dout.access_ptr('w'), shape_out[0]*shape_out[1], 
                            dtype_bytes(dtype_out), num, mode)

                def comp():
                    irb = tvm.ir_builder.create()
                    irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                    irb.emit(make_intrin_call("void", intrin_func,
                            dout.access_ptr('rw'),
                            din.access_ptr('r'),
                            dout.access_ptr('rw'),
                            shape_out[0]*shape_out[1],
                            self.get_mode_code(mode)
                            ))
                
                    return irb.get()
                return None, init, comp()
            
            return tvm.decl_tensor_intrin(out.op, lower_func, name=name, 
                                          binds={in1: in_buf,
                                                 out: out_buf})
        self.intrin_ctors['MAddMerge'] = mat_merge
        self.intrin_ctors['MMulMerge'] = mat_merge

        def mat_reduce_row(intrin_op, shape, scope_in='uni', scope_out='uni', mode='inc'):
            env = self.env
            cfg = self.env.cfg

            scope_in = self.get_scope(scope_in)
            scope_out = self.get_scope(scope_out, include_acc=True)

            dtype_in, dtype_out = self.mode2dtype(mode)

            # TODO: validate shape with cfg
            assert len(shape) == 2, 'the length of shape should be 2'
            nRow, nCol = shape
            
            # the name should contain all parameters
            name = intrin_op + ';' + str(nRow) + '_' + str(nCol) + '_' \
                    + scope_in + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            op_in = tvm.placeholder(shape, dtype_in, 'in')
            
            def expr_template(x, func, k):
                if (mode == 'inc'):
                    #x = x.
                    return lambda i: func(x[i, k].astype(dtype_out), k)
                elif (mode == 'dec'):
                    return lambda i: func(x[i, k], k).astype(dtype_out)
                else:
                    return lambda i: func(x[i, k], k)
            
            k = tvm.reduce_axis((0, nCol), 'k')
            if (intrin_op == 'MReduceSumRow'):
                expr = expr_template(op_in, tvm.sum, k)
                intrin_func = 'MReduceSumRow'
            else:
                raise ValueError('unsupported mat reduce row op')
            
            out = tvm.compute((nRow, ), expr, 'out')

            in_buf = self.decl_buffer(op_in, scope_in, 'in_buf', strides=(tvm.var('s'), 1))
            out_buf = self.decl_buffer(out, scope_out, 'out_buf')
            
            def lower_func(ins, outs):
                din1 = ins[0]
                dout = outs[0]

                init = self.emit_acc_init(dout.access_ptr('w'), 1, nRow, 0, 
                                mode, 0.0)

                def calc(toAccBuf, doAcc):
                    irb = tvm.ir_builder.create()
                    irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                    ptr_mode = 'rw' if doAcc else 'w'
                    irb.emit(make_intrin_call("void", intrin_func,
                                dout.access_ptr(ptr_mode),
                                din1.access_ptr('r'), 
                                din1.strides[0] * dtype_bytes(dtype_in),
                                shape[0], shape[1],
                                self.get_mode_code(mode),
                                toAccBuf, doAcc
                                ))

                    return irb.get()
                
                if (scope_out == env.acc_scope):
                    return calc(True, False), init, calc(True, True)
                else:
                    return calc(False, False)

            return tvm.decl_tensor_intrin(out.op, lower_func,
                                          name=name,
                                          binds={op_in: in_buf,
                                                 out: out_buf})
        self.intrin_ctors['MReduceSumRow'] = mat_reduce_row

        def mat_vctr_row(intrin_op, shape, scope_in_mat='uni', scope_in_vctr='uni', 
                         scope_out='uni', mode='n'):
            env = self.env
            cfg = self.env.cfg

            scope_in_mat = self.get_scope(scope_in_mat)
            scope_in_vctr = self.get_scope(scope_in_vctr)
            scope_out = self.get_scope(scope_out)

            dtype_in, dtype_out = self.mode2dtype(mode)

            # TODO: validate shape with cfg
            assert len(shape) == 2, 'the length of shape should be 2'
            nRow, nCol = shape
            
            # the name should contain all parameters
            name = intrin_op + ';' + str(nRow) + '_' + str(nCol) + '_' \
                    + scope_in_mat + ';' + scope_in_vctr + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            mat_in = tvm.placeholder(shape, dtype_in, 'mat_in')
            vctr_in = tvm.placeholder((nCol, ), dtype_in, 'vctr_in')

            def expr_template(mat, vctr, func):
                if (mode == 'inc'):
                    return lambda i, j: func(mat[i, j].astype(dtype_out), vctr[j])
                elif (mode == 'dec'):
                    return lambda i, j: func(mat[i, j], vctr[j]).astype(dtype_out)
                else:
                    return lambda i, j: func(mat[i, j], vctr[j])
            
            if (intrin_op == 'MAddV'):
                expr = expr_template(mat_in, vctr_in, lambda x, y: x + y)
                intrin_func = 'MAddV'
            elif (intrin_op == 'MSubV'):
                expr = expr_template(mat_in, vctr_in, lambda x, y: x - y)
                intrin_func = 'MSubV'
            elif (intrin_op == 'MMulV'):
                expr = expr_template(mat_in, vctr_in, lambda x, y: x * y)
                intrin_func = 'MMulV'
            else:
                raise ValueError('unsupported mat vctr intrin op')

            out = tvm.compute(shape, expr, 'out')
            mat_buf = self.decl_buffer(mat_in, scope_in_mat, 'mat_buf', strides=[tvm.var('s1'), 1])
            vctr_buf = self.decl_buffer(vctr_in, scope_in_vctr, 'in_buf')
            out_buf = self.decl_buffer(out, scope_out, 'out_buf', strides=[tvm.var('s2'), 1])

            def lower_func(ins, outs):
                ins = self.get_ins(ins, 'mat_buf', 'in_buf')
                din1, din2 = ins[0], ins[1]
                dout = outs[0]

                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(make_intrin_call("void", intrin_func,
                            dout.access_ptr('w', 'uint32'),
                            dout.strides[0] * dtype_bytes(dtype_out),
                            din1.access_ptr('r', 'uint32'),
                            din1.strides[0] * dtype_bytes(dtype_in),
                            din2.access_ptr('r', 'uint32'),
                            shape[0], shape[1],
                            self.get_mode_code(mode)
                            ))
                
                return irb.get()

            return tvm.decl_tensor_intrin(out.op, lower_func,
                                          name=name,
                                          binds={mat_in: mat_buf,
                                                 vctr_in: vctr_buf,
                                                 out: out_buf})
        
        self.intrin_ctors['MAddV'] = mat_vctr_row
        self.intrin_ctors['MSubV'] = mat_vctr_row
        self.intrin_ctors['MMulV'] = mat_vctr_row

        def mat_row_dot(intrin_op, shape, scope_in1='uni', scope_in2='uni', scope_out='uni',
                       mode='inc'):
            env = self.env
            cfg = self.env.cfg

            scope_in1 = self.get_scope(scope_in1)
            scope_in2 = self.get_scope(scope_in2)
            scope_out = self.get_scope(scope_out, include_acc=True)

            dtype_in, dtype_out = self.mode2dtype(mode)

            # TODO: validate shape with cfg
            assert len(shape) == 2, 'the length of shape should be 2'
            nRow, nCol = shape
            
            # the name should contain all parameters
            name = intrin_op + ';' + str(nRow) + '_' + str(nCol) + '_' \
                    + scope_in1 + ';' + scope_in2 + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]
            
            in1 = tvm.placeholder(shape, dtype_in, 'in1')
            in2 = tvm.placeholder(shape, dtype_in, 'in2')
            
            k = tvm.reduce_axis((0, nCol), 'k')
            if (mode == 'inc'):
                expr = lambda i: tvm.sum(in1[i, k].astype(dtype_out) * in2[i, k].astype(dtype_out), k)
            elif (mode == 'dec'):
                expr = lambda i: tvm.sum(in1[i, k] * in2[i, k], k).astype(dtype_out)
            else:
                expr = lambda i: tvm.sum(in1[i, k] * in2[i, k], k)
            out = tvm.compute((nRow, ), expr, 'out')

            in1_buf = self.decl_buffer(in1, scope_in1, 'in1_buf', strides=[tvm.var('s1'), 1])
            in2_buf = self.decl_buffer(in2, scope_in2, 'in2_buf', strides=[tvm.var('s2'), 1])
            out_buf = self.decl_buffer(out, scope_out, 'out_buf')
            
            def lower_func(ins, outs):
                din1, din2 = ins[0], ins[1]
                dout = outs[0]

                init = self.emit_acc_init(dout.access_ptr('w', 'uint32'),
                                    1, nRow, 0, mode, 0.0)

                def calc(toAccBuf, doAcc):
                    irb = tvm.ir_builder.create()
                    irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                    ptr_mode = 'rw' if doAcc else 'w'
                    irb.emit(make_intrin_call("void", 'MRowDot',
                                dout.access_ptr(ptr_mode, 'uint32'),
                                din1.access_ptr('r', 'uint32'),
                                din1.strides[0] * dtype_bytes(dtype_in),
                                din2.access_ptr('r', 'uint32'),
                                din2.strides[0] * dtype_bytes(dtype_in),
                                shape[0], shape[1],
                                self.get_mode_code(mode),
                                toAccBuf, doAcc
                                ))
                    
                    return irb.get()
                
                if (scope_out == env.acc_scope):
                    return calc(True, False), init, calc(True, True)
                else:
                    return calc(False, False)

            return tvm.decl_tensor_intrin(out.op, lower_func,
                                          name=name,
                                          binds={in1: in1_buf,
                                                 in2: in2_buf,
                                                 out: out_buf})
        self.intrin_ctors['MRowDot'] = mat_row_dot

        def vctr_sclr(intrin_op, scope_vctr='uni', scope_sclr='uni',
                      scope_out='uni', mode='n'):
            env = self.env
            cfg = self.env.cfg

            scope_vctr = self.get_scope(scope_vctr)
            scope_sclr = self.get_scope(scope_sclr)
            scope_out = self.get_scope(scope_out)

            dtype_in, dtype_out = self.mode2dtype(mode)

            # the name should contain all parameters
            name = intrin_op + scope_vctr + ';' + scope_sclr + ';' + scope_out + ';' + mode

            if (name in self.intrin_cache):
                return self.intrin_cache[name]

            shape = (cfg['vector_unit']['size'], )

            in1 = tvm.placeholder(shape, dtype_in, 'in1')
            in2 = tvm.placeholder((1, ), dtype_in, 'in2')

            def expr_template(x, y, func):
                if (mode == 'inc'):
                    return lambda i: func(x[i].astype(dtype_out), y[0].astype(dtype_out))
                elif (mode == 'dec'):
                    return lambda i: func(x[i], y[0]).astype(dtype_out)
                else:
                    return lambda i: func(x[i], y[0])

            if (intrin_op == 'VAddS'):
                expr = expr_template(in1, in2, lambda x, y: x + y)
                intrin_func = 'VAddS'
            elif (intrin_op == 'VSubS'):
                expr = expr_template(in1, in2, lambda x, y: x - y)
                intrin_func = 'VSubS'
            elif (intrin_op == 'VMulS'):
                expr = expr_template(in1, in2, lambda x, y: x * y)
                intrin_func = 'VMulS'
            elif (intrin_op == 'VDivS'):
                expr = expr_template(in1, in2, lambda x, y: x / y)
                intrin_func = 'VDivS'
            elif (intrin_op == 'VGTMS'):
                expr = expr_template(in1, in2, 
                                    lambda x, y: tvm.max(x, y))
                intrin_func = 'VGTMS'
            elif (intrin_op == 'SSubV'):
                expr = expr_template(in1, in2, lambda x, y: y - x)
                intrin_func = 'SSubV'
            elif (intrin_op == 'SDivV'):
                expr = expr_template(in1, in2, lambda x, y: y / x)
                intrin_func = 'SDivV'
            else:
                raise ValueError('unhandled intrin_op in vctr_binary')

            out = tvm.compute(shape, expr, 'out')
            in1_buf = self.decl_buffer(in1, scope_vctr, 'in1_buf')
            in2_buf = tvm.decl_buffer(in2.shape, in2.dtype, 'in2_buf',
                                      scope=scope_sclr, 
                                      data_alignment=dtype_bytes(in2.dtype), 
                                      offset_factor=1)
            out_buf = self.decl_buffer(out, scope_out, 'out_buf')
            
            def lower_func(ins, outs):
                ins = self.get_ins(ins, 'in1_buf', 'in2_buf')
                din1, din2 = ins[0], ins[1]
                dout = outs[0]
                irb = tvm.ir_builder.create()
                irb.scope_attr(env.nnpu_axis, "coproc_scope", 0)
                irb.emit(make_intrin_call("void", intrin_func,
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
        self.intrin_ctors['VAddS'] = vctr_sclr
        self.intrin_ctors['VSubS'] = vctr_sclr
        self.intrin_ctors['VMulS'] = vctr_sclr
        self.intrin_ctors['VDivS'] = vctr_sclr
        self.intrin_ctors['VGTMS'] = vctr_sclr
        self.intrin_ctors['SDivV'] = vctr_sclr
        self.intrin_ctors['SSubV'] = vctr_sclr

    def get(self, intrin_op, **kwargs):
        assert intrin_op in self.intrin_ctors, 'can not find constructor for intrin {0}'.\
            format(intrin_op)
        if ('mode' in kwargs):
            assert kwargs['mode'] in self.mode2code, \
                'illegal mode value {0}'.format(kwargs['mode'])
        return self.intrin_ctors[intrin_op](intrin_op, **kwargs)

    # convert scope name, also check whether scope is legal under current config
    def get_scope(self, scope_str, include_acc=False):
        return convert_scope(self.env, scope_str, include_acc=include_acc)
    
    def decl_buffer(self, tensor, scope, buf_name, **kwargs):
        dtype_bits = dtype_bytes(tensor.dtype) * 8
        return tvm.decl_buffer(
                tensor.shape, tensor.dtype, buf_name, scope=scope,
                data_alignment=self.env.scope2config(scope)['width_per_channel'] / 8,
                offset_factor=self.env.scope2config(scope)['width_per_channel'] / dtype_bits,
                **kwargs)

    def mode2dtype(self, mode):
        cfg = self.env.cfg
        assert mode in ['w', 'n', 'inc', 'dec'], 'invalid mode string'

        dtype_in = cfg['dtype_w'] if mode in ['w', 'dec'] else cfg['dtype_n']
        dtype_out = cfg['dtype_w'] if mode in ['w', 'inc'] else cfg['dtype_n']
        return dtype_in, dtype_out

    def get_mode_code(self, mode_str):
        return self.mode2code[mode_str]

    def get_ins(self, ins, *args):
        res = []
        for name in args:
            #print(name)
            for din in ins:
                if (din.name == name):
                    res.append(din)
        return res
    
    def emit_memset(self, addr, nUnit, stride, val, mode):
        if (not val is tvm.expr.FloatImm):
            val = tvm.const(val, 'float64')
        irb = tvm.ir_builder.create()
        irb.scope_attr(self.env.nnpu_axis, "coproc_scope", 0)
        irb.emit(make_intrin_call("void", 'Memset',
                                addr, nUnit, stride,
                                val, self.get_mode_code(mode)
                    ))
        return irb.get()

    def emit_acc_init(self, addr, nRow, nCol, rowStride, mode, val=0.0):
        if (not val is tvm.expr.FloatImm):
            val = tvm.const(val, 'float64')
        irb = tvm.ir_builder.create()
        irb.scope_attr(self.env.nnpu_axis, "coproc_scope", 0)
        irb.emit(make_intrin_call("void", 'AccMemset',
                addr,
                rowStride,
                nRow, nCol,
                val, self.get_mode_code(mode)
                ))
        return irb.get()