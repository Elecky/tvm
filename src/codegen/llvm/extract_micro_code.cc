#include "extract_micro_code.h"
#include <sstream>
#include <tuple>
#include <tvm/arithmetic.h>
#include "../../arithmetic/compute_expr.h"
#include <tvm/ir_visitor.h>
#include <llvm/IR/Function.h>
#include "../../pass/ir_util.h"
#include <tvm/ir_functor_ext.h>
#include "../../arithmetic/split_expr.h"

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
 * \brief a exception to indicate that the mutator failed to expand the body of of micro-kernel.
*/
class cannot_expand_error : public std::exception {
public:
    cannot_expand_error(const string &message, bool _should_report = false) :
        message_(message),
        should_report_(_should_report)
    {}

    cannot_expand_error() :
        should_report_(false)
    {}

    const char * what() const noexcept final {
        return message_.c_str();
    }

    inline bool should_report() const noexcept {
        return should_report_;
    }
private:
    string message_;
    /* indicates whether this message should be reported to user/developer.
       currently, errors caused by bad config file should be reported */
    bool should_report_;
};

/**!
 * \brief print micro-kernel to asm format.
*/
class micro_code_asm_printer : IRVisitor {
public:
    micro_code_asm_printer(
            const unordered_map<string, string> *uop_templates,
            unsigned num_loop_levels) :
        uop_templates_(uop_templates),
        num_loop_levels_(num_loop_levels),
        micro_code_count(0)
    {}

    pair<unsigned, string> print(const Stmt &s) {
        Visit(s);
        return {micro_code_count, outs.str()};
    }

    void Visit_(const Call *op) final;

private:
    const unordered_map<string, string> *uop_templates_;
    unsigned num_loop_levels_;

    stringstream outs;

    /* the number of micro codes generated. */
    unsigned micro_code_count;

    template <typename T>
    inline auto get_value(const Expr &expr) -> decltype(T::value) {
        const T *node = expr.as<T>();
        if (node != nullptr) {
            return node->value;
        }
        else {
            throw cannot_expand_error(string("when printing NNPU micro-code asm, wrong argument type met, real type is: ") + expr->type_key(), true);
            return decltype(T::value)();
        }
    }
};

void micro_code_asm_printer::Visit_(const Call *op) {
    if (op->call_type != Call::Intrinsic && op->call_type != Call::PureIntrinsic) {
        throw cannot_expand_error("when trying to print micro-code asm of NNPU, non-Intrinsic call met", true);
        return;
    }
    
    /* get the call intrinsic name. */
    auto it = uop_templates_->find(op->name);
    if (it == uop_templates_->end()) {
        string error = "when trying to print micro-code asm of NNPU, corresponding uop template was not found for call: ";
        error.append(op->name);
        throw cannot_expand_error(error, true);
    }

    /* first print the uop name. uop name is exactly the call->name */
    outs << op->name << ' ';
    /* then follows the template to print arguments */

    const string &temp = it->second;
    const auto &args = op->args;
    for (size_t idx = 0, arg_idx = 0; idx < temp.length(); ++idx) {
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
            throw cannot_expand_error("invalid uop template: " + temp, true);
            break;
        }
    }

    /* print a new line character. */
    outs << '\n';
    ++micro_code_count;
}

/**!
 * \brief expand the body of a micro code.
*/
class micro_code_expander : IRMutator {
public:
    micro_code_expander(
            Array<Var> loop_vars, unsigned num_loop_levels, 
            const unordered_map<string, string> *uop_templates) :
        loop_vars_(loop_vars),
        uop_templates_(uop_templates),
        num_loop_levels_(num_loop_levels),
        input_exprs(1)  // initialize first expr (index 0) to empty.
    {}

    /**!
     * \brief
    */
    std::tuple<bool, string, std::vector<Expr>, unsigned> expand(const Stmt &s);

    Stmt Mutate_(const For *op, const Stmt &s) final;

    /**!
     * \brief subroutine used by Mutate_(For *, xxx), unroll the for loop.
    */
    Stmt unroll(const For *op, const Stmt &s);

    Expr Mutate_(const Call *op, const Expr &expr) final;

private:
    /* micro-code loop variables */
    Array<Var> loop_vars_;
    /* templates of uop instructions. */
    const unordered_map<string, string> *uop_templates_;
    /* the loop variables of loops in micro-code, those loops are to be expanded. 
       actually, they are loop vars of loops on the tree path. */
    vector<Var> expand_vars;
    /* the number of loop levels of each micro-kernel. */
    unsigned num_loop_levels_;

    /* expressions that should be computed by ALU, and then passed to micro-kernel runner. */
    vector<Expr> input_exprs;

    /**!
     * \brief 
    */
    static bool detect_imm_linear_equation(
            const Expr &expr, const Array<Var> &vars,
            Array<Expr> &coef_imm, Expr &base, unsigned pad = 0);

    /**!
     * \brief 
    */
    inline static bool split_expr(
            const Expr &expr, const Array<Var> &vars,
            Expr &expr1, Expr &expr2) {
        std::unordered_set<const Variable*> var_set;
        for (auto &item : vars) {
            var_set.insert(item.get());
        }

        return SplitExpr(expr, var_set, expr1, expr2);
    }

    /**!
     * \brief allocate a pipeline register to EXPR.
    */
    inline int allocate_register(const Expr &expr) {
        if (is_zero(expr)) {
            return 0;
        }
        // TODO: use deep comparasion to compare already allocated expression to current one.
        input_exprs.push_back(expr);
        return static_cast<int>(input_exprs.size() - 1);
    }

    inline int GetExtent(const For* op) {
        // constant folding.
        Expr extent = ir::Simplify(op->extent);
        
        if (const auto *ptr = as_const_int(extent)) {
            return static_cast<int>(*ptr);
        }
        if (const auto *ptr = as_const_uint(extent)) {
            return static_cast<int>(*ptr);
        }
        return -1;
    }
};

std::tuple<bool, string, std::vector<Expr>, unsigned> 
micro_code_expander::expand(const Stmt &s) {
    using std::make_tuple;

    try {
        Stmt expanded = Mutate(s);
        /* simplify expanded stmt, making sure all constants are folded. */
        expanded = Simplify(expanded);
        
        micro_code_asm_printer printer(uop_templates_, num_loop_levels_);
        auto res = printer.print(expanded);
        return make_tuple(true, move(res.second), move(input_exprs), res.first);
    }
    catch (cannot_expand_error &error) {
        if (error.should_report()) {
            LOG(WARNING) << error.what();
        }
        return make_tuple(false, "", vector<Expr>(), 0);
    }
}

Stmt micro_code_expander::Mutate_(const For *op, const Stmt &s) {
    /* first push current loop variable into EXPAND_VARS, then call IRMutator::mutate_ to mutate body. */
    expand_vars.push_back(Var(op->loop_var.node_));
    Stmt stmt = IRMutator::Mutate_(op, s);
    // pop the var.
    expand_vars.pop_back();

    op = stmt.as<For>();
    /* then we try to unroll the for loop. */
    return unroll(op, stmt);
}

Stmt micro_code_expander::unroll(const For *op, const Stmt &s) {
    using arith::ComputeExpr;
    int value = GetExtent(op);
    // For loop must have a constant integer extent
    if (value == -1) {
        throw cannot_expand_error("non-constant loop extent met");
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

Expr micro_code_expander::Mutate_(const Call *op, const Expr &expr) {
    if (op->call_type != Call::Intrinsic && op->call_type != Call::PureIntrinsic) {
        throw cannot_expand_error("when trying to expand micro-code of NNPU, non-Intrinsic call met", true);
    }
    
    /* get the call intrinsic name. */
    auto it = uop_templates_->find(op->name);
    if (it == uop_templates_->end()) {
        string error = "when trying to expand micro-code of NNPU, corresponding uop template was not found for call: ";
        error.append(op->name);
        throw cannot_expand_error(error, true);
    }
    
    if (it->second.length() != op->args.size()) {
        std::stringstream error;
        error << "when trying to expand micro-code of NNPU, number of args in uop template and call doesn't match, template: "
                << it->second << ", while " << op->args.size() << " arguments in call";
        throw cannot_expand_error(error.str(), true);
    }
    
    /* the converted arguments. */
    Array<Expr> args;
    /* loop variable of loops to be unrolled. */
    Array<Var> unroll_vars(expand_vars.begin(), expand_vars.end());
    /* handle every argument. */
    for (size_t idx = 0; idx < it->second.length(); ++idx) {
        /* the argument type at index IDX of this uop instruction. */
        char type = it->second[idx];
        /* the argument expression. */
        Expr arg = op->args[idx];
        if (!arg.defined()) {
            throw cannot_expand_error("an undefined argument met in Call", true);
        }
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
                // otherwise, it means the call doesn't satify uop definition, so can't convert it to uop.
                throw cannot_expand_error("non-const argument met, when template indicates it should be const");
            }
            break;

        case 'f':  // 'f' means a float immediate value argument in uop call.
            arg = Simplify(arg);
            if (arg->is_type<FloatImm>()) {
                args.push_back(arg);
            }
            else {
                throw cannot_expand_error("non-const argument met, when template indicates it should be const");
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
                throw cannot_expand_error("failed to detect the coefficient of loop variable from composite operand expression");
            }
            
            /* now check if base = f(unroll_vars)+tail, where TAIL is invariant to UNROLL_VARS, and f only depends on UNROLL_VARS */
            Expr f_unrool_vars, tail;
            if (!split_expr(base, unroll_vars, f_unrool_vars, tail)) {
                throw cannot_expand_error("failed to split free variable from expression");
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
            throw cannot_expand_error("invalid uop template: " + it->second, true);
        }
    }
    return Call::make(op->type, op->name, args, op->call_type);
}

bool micro_code_expander::detect_imm_linear_equation(
        const Expr &expr, const Array<Var> &vars,
        Array<Expr> &coef_imm, Expr &base, unsigned pad) {
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
        if (coef.defined() && coef->is_type<IntImm>()) {
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

    inline pair<Stmt, vector<string>> fold(const Stmt &s) {
        Stmt ret = Mutate(s);
        return {ret, move(micro_kernels)};
    }

    Stmt Mutate_(const For *op, const Stmt &s);

    Expr Mutate_(const Call *op, const Expr &s) final {
        /* the call should have been handled in Mutate(Evaluate*, Stmt) method. */
        throw std::logic_error("unexpected Node Call");
    }

    Stmt Mutate_(const Evaluate *op, const Stmt &s) final;

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

    Stmt build_launch_kernel(const Array<Expr> loop_exts, const vector<Expr> &local_reg_vals);
};

Stmt loop_folder::Mutate_(const For *op, const Stmt &s) {
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
    // TODO: I limited the micro-code count to be less than 256, is there any beeter way to do limitation?
    if (std::get<0>(res) && std::get<3>(res) < 256) {
        /* insert this micro-kernel into list. */
        micro_kernels.push_back(move(std::get<1>(res)));
        /* build statements to set local register and launch micro-kernel */
        return build_launch_kernel(loop_exts, std::get<2>(res));
    }
    else {
        /* if fails, forward to IRMutator to skip this level of loop, continue to try expanding inner loops to generate micro-kernel body. */
        return IRMutator::Mutate_(op, s);
    }
}

Stmt loop_folder::Mutate_(const Evaluate *op, const Stmt &s) {
    /* handle the condition that a micro-code is not wrapped in any loop. */
    micro_code_expander expander(Array<Var>()/* no loop var */, 2, uop_templates_);
    auto res = expander.expand(s);
    if (std::get<0>(res)) {
        /* insert this micro-kernel into list. */
        micro_kernels.push_back(move(std::get<1>(res)));
        /* build statements to set local register and launch micro-kernel */
        Array<Expr> loop_exts;
        loop_exts.push_back(make_const(Int(32), 1));
        loop_exts.push_back(make_const(Int(32), 1));
        return build_launch_kernel(loop_exts, std::get<2>(res));
    }
    else {
        throw std::logic_error("failed to generate kernel for a single Call Node");
    }
}

Stmt loop_folder::build_launch_kernel(const Array<Expr> loop_exts, const vector<Expr> &local_reg_vals) {
    /* current pipeline id as Expr */
    Expr pid = make_const(Int(32), pipeline_id_);
    /* list of stmts to execute */
    vector<Stmt> stmts;
    /* first make intrinsic calls to set pipeline local registers */
    for (size_t idx = 1; idx < local_reg_vals.size(); ++idx) {
        Array<Expr> args;
        args.push_back(pid);  // arg0: pipeline id
        args.push_back(make_const(Int(32), static_cast<int64_t>(idx)));  // arg1: local register no.
        args.push_back(local_reg_vals[idx]);  // arg2: the value.
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
        if (const auto *ptr = as_const_int(op->value)) {
            int pipeline_id = *ptr;

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