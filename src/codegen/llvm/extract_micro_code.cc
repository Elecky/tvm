#include "extract_micro_code.h"
#include <sstream>
#include <tuple>
#include <tvm/arithmetic.h>
#include "../../arithmetic/compute_expr.h"
#include <tvm/ir_visitor.h>
#include <llvm/IR/Function.h>
#include "../../pass/ir_util.h"

namespace tvm {
namespace ir {

using std::move;
using std::vector;
using std::string;
using std::unordered_map;
using std::stringstream;
using std::pair;

using namespace tvm::arith;

Stmt call_llvm_intrin_with_side_effect(Type rtype, const string &name, int num_signature, bool has_return, const Array<Expr> &args) {
    auto intrin_id = llvm::Function::lookupIntrinsicID(name);
    Array<Expr> full_args;
    full_args.push_back( make_const(UInt(32), intrin_id) );  // arg0: intrinsic function id
    full_args.push_back( make_const(UInt(32), num_signature) );  // arg1: num_signature
    full_args.push_back( has_return ? make_const(UInt(32), 1) : make_zero(UInt(32)) );  // arg2: has_return?
    for (auto &item : args) {
        full_args.push_back(item);
    }

    return Evaluate::make(Call::make(rtype, "llvm_intrin_with_side_effct", full_args, Call::CallType::Intrinsic));
}

Stmt call_void_llvm_intrin(const string &name, const Array<Expr> &args) {
    return call_llvm_intrin_with_side_effect(Int(32), name, 0, false, args);
}

/**!
 * \brief 
*/
class micro_code_asm_printer : IRVisitor {
public:
    micro_code_asm_printer(
            const unordered_map<string, string> *uop_templates,
            unsigned num_loop_levels) :
        uop_templates_(uop_templates),
        num_loop_levels_(num_loop_levels),
        fail(false)
    {}

    pair<bool, string> print(const Stmt &s) {
        Visit(s);
        if (!fail) {
            return {true, outs.str()};
        }
        else {
            return {false, ""};
        }
    }

    void Visit_(const Call *op) final {
        if (fail) {
            // early stop.
            return;
        }

        if (op->call_type != Call::Intrinsic && op->call_type != Call::PureIntrinsic) {
            fail_with_info("when trying to print micro-code asm of NNPU, non-Intrinsic call met");
            return;
        }
        
        /* get the call intrinsic name. */
        auto it = uop_templates_->find(op->name);
        if (it == uop_templates_->end()) {
            string error = "when trying to print micro-code asm of NNPU, corresponding uop template was not found for call: ";
            error.append(op->name);
            fail_with_info(error);
            return;
        }

        /* first print the uop name. uop name is exactly the call->name */
        outs << op->name << ' ';
        /* then follows the template to print arguments */

        const string &temp = it->second;
        const auto &args = op->args;
        for (size_t idx = 0, arg_idx = 0; idx < temp.length(); ++idx) {
            if (fail) {
                break;
            }
            /* print a comma if this is not first argument. */
            if (idx != 0) {
                outs << ", ";
            }

            char type = temp[idx];
            switch (type)
            {
            case 'i':
                CHECK_LT(arg_idx, args.size()) << ", too small arguments than that specified in template";
                outs << get_value<IntImm>(args[arg_idx++]);
                break;

            case 'u':
                CHECK_LT(arg_idx, args.size()) << ", too small arguments than that specified in template";
                outs << get_value<UIntImm>(args[arg_idx++]);
                break;
            
            case 'f':
                CHECK_LT(arg_idx, args.size()) << ", too small arguments than that specified in template";
                outs << get_value<FloatImm>(args[arg_idx++]);
                break;

            case 'c':
                CHECK_LT(arg_idx + num_loop_levels_ + 1, args.size()) << ", too small arguments than that specified in template";
                /* the format of a composite argument is:
                   {iter_var0_coef:iter_var1_coef:...:addend:local_reg} */
                outs << '{';
                for (size_t coef_idx = 0; coef_idx < num_loop_levels_; ++coef_idx) {
                    outs << get_value<IntImm>(args[arg_idx++]) << ':';
                }
                outs << get_value<IntImm>(args[arg_idx++]) << ':';
                outs << get_value<IntImm>(args[arg_idx++]) << '}';
                break;

            default:
                fail_with_info("invalid uop template: " + temp);
                break;
            }
        }

        /* print a new line character. */
        outs << '\n';
    }

private:
    const unordered_map<string, string> *uop_templates_;
    unsigned num_loop_levels_;

    stringstream outs;

    /* did any error already occur? that is, expansion can not be finished.
       this may be caused by a unsupported expression pattern in body, or pipeline register number limitation exceeded. */
    bool fail;

    inline void fail_with_info(const string &info) {
        fail = true;
        if (info.length() != 0) {
            LOG(WARNING) << info;
        }
    }

    template <typename T>
    auto get_value(const Expr &expr) -> decltype(T::value) {
        const T *node = expr.as<T>();
        if (node != nullptr) {
            return node->value;
        }
        else {
            fail_with_info(string("when printing NNPU micro-code asm, wrong argument type met, real type is: ") + expr->type_key());
            return decltype(T::value)();
        }
    }
};

/**!
 * \brief expand the body of a micro code.
*/
class micro_code_expander : IRMutator {
public:
    micro_code_expander(
            Array<Var> loop_vars, unsigned num_loop_levels, 
            const unordered_map<string, string> *uop_templates) :
        loop_vars_(loop_vars),
        fail(false),
        uop_templates_(uop_templates),
        num_loop_levels_(num_loop_levels),
        input_exprs(1)  // initialize first expr (index 0) to empty.
    {}

    std::tuple<bool, string, std::vector<Expr>> expand(const Stmt &s) {
        using std::make_tuple;

        fail = false;
        Stmt expanded = Mutate(s);
        if (fail) {
            return make_tuple(false, "", vector<Expr>());
        }
        /* simplify expanded stmt, making sure all constants are folded. */
        expanded = Simplify(expanded);
        
        micro_code_asm_printer printer(uop_templates_, num_loop_levels_);
        auto res = printer.print(expanded);
        if (!res.first) {
            return make_tuple(false, "", vector<Expr>());
        }

        return make_tuple(true, move(res.second), move(input_exprs));
    }

    Stmt Mutate_(const For *op, const Stmt &s) final {
        /* first push current loop variable into EXPAND_VARS, then call IRMutator::mutate_ to mutate body. */
        expand_vars.push_back(Var(op->loop_var.node_));
        Stmt stmt = IRMutator::Mutate_(op, s);
        expand_vars.pop_back();

        if (!fail) {
            op = stmt.as<For>();
            return unroll(op, stmt);
        }
        else {
            return s;
        }
    }

    Stmt unroll(const For *op, const Stmt &s) {
        using arith::ComputeExpr;
        int value = GetExtent(op);
        // For loop must have a constant integer extent
        if (value == -1) {
            fail = true;
            return s;
        }
        if (value == 0) return Evaluate::make(0);
        Stmt body = op->body;
        Map<Var, Expr> vmap;
        Var lv(op->loop_var.node_);
        Stmt unrolled;
        for (int i = 0; i < value; ++i) {
            vmap.Set(lv,
                    ComputeExpr<Add>(
                            op->min, make_const(op->loop_var.type(), i)));
            Stmt step = Substitute(body, vmap);
            if (unrolled.defined()) {
                unrolled = Block::make(unrolled, step);
            } else {
                unrolled = step;
            }
        }
        return unrolled;
    }

    Expr Mutate_(const Call *op, const Expr &expr) final {
        if (op->call_type != Call::Intrinsic && op->call_type != Call::PureIntrinsic) {
            fail_with_info("when trying to expand micro-code of NNPU, non-Intrinsic call met");
            return expr;
        }
        
        /* get the call intrinsic name. */
        auto it = uop_templates_->find(op->name);
        if (it == uop_templates_->end()) {
            string error = "when trying to expand micro-code of NNPU, corresponding uop template was not found for call: ";
            error.append(op->name);
            fail_with_info(error);
            return expr;
        }
        
        if (it->second.length() != op->args.size()) {
            std::stringstream error;
            error << "when trying to expand micro-code of NNPU, number of args in uop template and call doesn't match, template: "
                    << it->second << ", while " << op->args.size() << " arguments in call";
            fail_with_info(error.str());
            return expr;
        }

        /* the converted arguments. */
        Array<Expr> args;
        /* loop variable of loops to be unrolled. */
        Array<Var> unroll_vars(expand_vars.begin(), expand_vars.end());
        /* handle every argument. */
        for (size_t idx = 0; idx < it->second.length(); ++idx) {
            if (fail) {
                break;
            }
            /* the argument type at index IDX of this uop instruction. */
            char type = it->second[idx];
            /* the argument expression. */
            Expr arg = op->args[idx];

            switch (type)
            {
            case 'i':  // 'i' means a integer immediate value argument in uop call.
                /* fall through */
            case 'u':
                arg = Simplify(arg);
                if (   (type == 'i' && arg->is_type<IntImm>()) 
                    || (type == 'u' && arg->is_type<UIntImm>()) ) {
                    /* if this argument is immediate value, add it to converted arguments. */
                    args.push_back(arg);
                }
                else {
                    fail = true;  // otherwise, it means the call doesn't satify uop definition, so can't convert it to uop.
                }
                break;

            case 'f':  // 'f' means a float immediate value argument in uop call.
                arg = Simplify(arg);
                if (arg->is_type<FloatImm>()) {
                    args.push_back(arg);
                }
                else {
                    fail = true;
                }
                break;

            case 'c':
            {
                // 'c' means
                /* COEF_IMM is the coefficient of loop_vars, they are integer immediates. */
                Array<Expr> coef_imm;
                /* base is the expression after subtracting COEF*LOOP_VARS_ from ARG */
                Expr base;
                if (!detect_imm_linear_equation(arg, loop_vars_, coef_imm, base, num_loop_levels_ - loop_vars_.size())) {
                    fail = true;
                    break;
                }
                
                /* now check if base = f(unroll_vars)+tail, where TAIL is invariant to UNROLL_VARS, and f only depends on UNROLL_VARS */
                Expr f_unrool_vars, tail;
                if (!split_expr(base, unroll_vars, f_unrool_vars, tail)) {
                    fail = true;
                    break;
                }
                /* allocate a pipeline local register to store the expression TAIL. the expression will be computed by ALU. */
                int tail_reg = allocate_register(tail);
                /* if all passed, we insert arguments. the order is {coef_imm, f(unroll_vars), tail_reg} */
                for (const auto &item : coef_imm) {
                    args.push_back(item);
                }
                args.push_back(f_unrool_vars);
                args.push_back(IntImm::make(Int(32), tail_reg));

                break;
            }
            default:
                fail_with_info("invalid uop template: " + it->second);
                break;
            }
        }

        if (fail) {
            return expr;
        }
        else {
            return Call::make(op->type, op->name, args, op->call_type);
        }
    }

private:
    /* micro-code loop variables */
    Array<Var> loop_vars_;
    /* did any error already occur? that is, expansion can not be finished.
       this may be caused by a unsupported expression pattern in body, or pipeline register number limitation exceeded. */
    bool fail;
    /* templates of uop instructions. */
    const unordered_map<string, string> *uop_templates_;
    /* the loop variables of loops in micro-code, those loops are to be expanded. 
       actually, they are loop vars of loops on the tree path. */
    vector<Var> expand_vars;
    /* the number of loop levels of each micro-kernel. */
    unsigned num_loop_levels_;

    /* expressions that should be computed by ALU, and then passed to micro-kernel runner. */
    vector<Expr> input_exprs;

    inline void fail_with_info(const string &info) {
        fail = true;
        if (info.length() != 0) {
            LOG(WARNING) << info;
        }
    }

    /**!
     * \brief 
    */
    static bool detect_imm_linear_equation(
            const Expr &expr, const Array<Var> &vars,
            Array<Expr> &coef_imm, Expr &base, unsigned pad = 0) {
        /* first try to detect linear relationship to loop_vars_ */
        Array<Expr> coefs;
        if (vars.size() != 0) {
            coefs = DetectLinearEquation(expr, vars);
        }
        else {
            coefs.push_back(expr);
        }
        if (coefs.size() == 0) {
            // if the argument is not linear to loop_vars_.
            return false;
        }
        // add zeros before real coefficients.
        coef_imm = Array<Expr>(pad, make_zero(Int(32)));
        // then check whether the coefficients are all integer immediates.
        for (size_t idx = 0; idx < coefs.size() - 1; ++idx) {
            auto coef = Simplify(coefs[idx]);
            if (coef->is_type<IntImm>()) {
                coef_imm.push_back(coef);
            }
            else {
                // if the coefficient is not int immediate, then fails.
                return false;
            }
        }
        base = Simplify(*coefs.rbegin());
        return true;
    }

    /**!
     * \brief 
    */
    static bool split_expr(
            const Expr &expr, const Array<Var> &vars,
            Expr &expr1, Expr &expr2) {
        // TODO: implement real expression splitter to replace DetectLinearEquation!!
        Array<Expr> coef;
        if (detect_imm_linear_equation(expr, vars, coef, expr2)) {
            /* now construct expr1 */
            Expr part1 = make_zero(expr.type());
            for (size_t idx = 0; idx < vars.size(); ++idx) {
                /* part1 = part1+coef[idx]*vars[idx] */
                part1 = part1 + Mul::make( coef[idx], vars[idx] );
            }

            if (expr2->is_type<IntImm>()) {
                expr1 = part1 + expr2;
                expr2 = make_zero(Int(32));
            }
            else {
                expr1 = part1;
            }
            return true;
        }
        else {
            return false;
        }
    }

    /**!
     * \brief allocate a pipeline register to EXPR.
    */
    int allocate_register(const Expr &expr) {
        if (const IntImm * imm = expr.as<IntImm>()) {
            if (imm->value == 0) {
                return 0;
            }
        }
        // TODO: use deep comparasion to compare already allocated expression to current one.
        input_exprs.push_back(expr);
        return static_cast<int>(input_exprs.size() - 1);
    }

    int GetExtent(const For* op) {
        // constant folding.
        Expr extent = ir::Simplify(op->extent);
        
        if (const IntImm *v1 = extent.as<IntImm>()) {
            return static_cast<int>(v1->value);
        }
        if (const UIntImm *v2 = extent.as<UIntImm>()) {
            return static_cast<int>(v2->value);
        }
        return -1;
    }
};

/**!
 * \brief extractor that generates micro codes for one pipeline.
*/
class loop_folder : public IRMutator {
public:
    /**!
     * \brief constructor a loop_folder.
     * \param start_idx: all micro codes will be numbered from START_IDX
     * \param pipeline_id: the id of pipeline to launch micro code.
    */
    loop_folder(unsigned start_idx, int pipeline_id,
                const unordered_map<string, string> *uop_templates) :
        start_idx_(start_idx),
        pipeline_id_(pipeline_id),
        uop_templates_(uop_templates)
    {}

    pair<Stmt, vector<string>> fold(const Stmt &s) {
        Stmt ret = Mutate(s);
        return {ret, move(micro_kernels)};
    }

    Stmt Mutate_(const For *op, const Stmt &s) final {
        Array<Var> loop_vars;
        Stmt body;
        /* extend of loops */
        Array<Expr> loop_exts;
        // TODO: check the lower bound of loop(s)

        if (const For *inner_op = op->body.as<For>()) {
            /* if there are two for loops */
            Var iter_var0(op->loop_var.node_);
            loop_vars.push_back(iter_var0);
            Var iter_var1(inner_op->loop_var.node_);
            loop_vars.push_back(iter_var1);
            body = inner_op->body;

            loop_exts.push_back(op->extent);
            loop_exts.push_back(inner_op->extent);
        }
        else {
            Var iter_var(op->loop_var.node_);
            loop_vars.push_back(iter_var);
            body = op->body;

            /* if there are only one loop, we set the extent of outter loop to 1 */
            loop_exts.push_back(make_const(Int(32), 1));
            loop_exts.push_back(op->extent);
        }

        micro_code_expander expander(loop_vars, 2, uop_templates_);
        auto res = expander.expand(body);

        if (std::get<0>(res)) {
            /* current pipeline id as Expr */
            Expr pid = make_const(Int(32), pipeline_id_);
            /* list of stmts to execute */
            vector<Stmt> stmts;

            /* insert this micro-kernel into list. */
            micro_kernels.push_back(move(std::get<1>(res)));

            /* first make intrinsic calls to set pipeline local registers */
            const vector<Expr> &local_regs_vals = std::get<2>(res);
            for (size_t idx = 1; idx < local_regs_vals.size(); ++idx) {
                Array<Expr> args;
                args.push_back(pid);  // arg0: pipeline id
                args.push_back(make_const(Int(32), static_cast<int64_t>(idx)));  // arg1: local register no.
                args.push_back(local_regs_vals[idx]);  // arg2: the value.
                stmts.push_back(call_void_llvm_intrin("llvm.NNPU.SetPipelineReg", args));
            }
            /* then make call to launch micro-kernel */
            Array<Expr> args;
            Expr kernel_id = make_const(Int(32), start_idx_ + micro_kernels.size() - 1);
            args.push_back(pid);  // arg0: pipeline id
            args.push_back(kernel_id);  // arg1: micro kernel id
            // loop extens.
            for (auto &item : loop_exts) {
                args.push_back(item);
            }
            stmts.push_back(call_void_llvm_intrin("llvm.NNPU.LaunchMicroKernel", args));

            return MergeSeq(stmts);
        }
        else {
            /* if fails, forward to IRMutator to skip this level of loop, continue to try expanding inner loops to generate micro-kernel body. */
            return IRMutator::Mutate_(op, s);
        }
    }

    Expr Mutate_(const Call *op, const Expr &s) final {
        throw std::logic_error("unexpected Node Call");
    }

    inline vector<string> & get_micro_codes() {
        return micro_kernels;
    }

private:
    vector<string> micro_kernels;
    /* number micro codes from START_IDX. */
    unsigned start_idx_;
    /* the ID of pipeline on which micro codes should be launched. */
    int pipeline_id_;
    /* templates of uop instructions. */
    const unordered_map<string, string> *uop_templates_;
};

micro_code_extractor::micro_code_extractor(
        const unordered_map<string, string> *uop_templates,
        unsigned start_idx,
        string attr_key) :
    start_idx_(start_idx),
    uop_templates_(move(uop_templates)),
    attr_key_(move(attr_key))
{}

Stmt micro_code_extractor::Mutate_(const AttrStmt* op, const Stmt& s) {
    if (op->attr_key == attr_key_) {
        if (const IntImm *value = op->value.as<IntImm>()) {
            int pipeline_id = value->value;

            /* the pass in start_idx = start_idx_ + current count of micro-kernels. */
            loop_folder folder(start_idx_ + micro_kernels.size(), pipeline_id, uop_templates_);
            auto res = folder.fold(op->body);

            /* add all micro-kernels in this subtree to list. */
            for (string &item : res.second) {
                micro_kernels.push_back(move(item));
            }

            return AttrStmt::make(op->node, attr_key_, op->value, res.first);
        }
        else {
            throw std::logic_error("value of coproc_uop_scope is not IntImm!");
        }
    }
    else {
        return IRMutator::Mutate_(op, s);
    }
}

std::pair<Stmt, std::vector<std::string> > micro_code_extractor::extract(const Stmt &body) {
    micro_kernels = vector<string>();

    Stmt mutated_body = Mutate(body);

    return {mutated_body, move(micro_kernels)};
}

pair<Array<LoweredFunc>, vector<std::string> >
generate_micro_kernels(const Array<LoweredFunc> &funcs, const std::unordered_map<std::string, std::string> &uop_templates) {
    Array<LoweredFunc> ret_funcs;
    vector<string> micro_kernels;

    for (const LoweredFunc & func : funcs) {
        micro_code_extractor extractor(&uop_templates, micro_kernels.size());

        auto res = extractor.extract(func->body);

        NodePtr<LoweredFuncNode> n = make_node<LoweredFuncNode>(*func.operator->());
        n->body = res.first;

        assert(n->func_type == kDeviceFunc);
        // put the converted device function into list.
        ret_funcs.push_back(LoweredFunc(n));
        // put the micro-kernels into list.
        for (string &item : res.second) {
            micro_kernels.push_back(move(item));
        }
    }

    return {ret_funcs, move(micro_kernels)};
}

}  // namespace ir
}  // namespace tvm