#include "split_expr.h"
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_operator.h>
#include <tvm/ir_pass.h>
#include "compute_expr.h"

namespace tvm {
namespace arith {

using namespace ir;

struct IndependEntry {
    /* see expr_spliter. depend_part is the sub-expression only contains var in V_SET, free_part doesn't contain variables in V_SET. 
       NOTE: free_part should never be immediate value. */
    Expr depend_part, free_part;
};

/**!
 * \brief given a set of variables V_SET, split an expression to sum of two expressions, namely, e1 + e2, 
 *        where e1 contiains only variables in V_SET, and e2 doesn't contain any variable in V_SET.
*/
class ExprSpliter : public ExprFunctor<IndependEntry (const Expr&, const Expr &)> {
public:
    ExprSpliter(std::unordered_set<const Variable*> vars) :
        vars_(move(vars)),
        fail(false)
    {}

    /**!
     * \brief given a set of variables V_SET, split an expression to sum of two expressions, namely, e1 + e2, 
     *        where e1 contiains only variables in V_SET, and e2 doesn't contain any variable in V_SET.
     * \param expr: the expression to split.
     * \param depend: return reference, the e1 in brief.
     * \param free: return reference, the e2 in brief.
     * \return does split success?
    */
    bool Split(const Expr &expr, Expr &depend, Expr &free);

    IndependEntry VisitExpr_(const Add *op, const Expr &e) final;
    IndependEntry VisitExpr_(const Sub *op, const Expr &e) final;
    IndependEntry VisitExpr_(const Mul *op, const Expr &e) final;
    IndependEntry VisitExpr_(const Variable *op, const Expr &e) final;
    // TODO: handle division.
    // IndependEntry VisitExpr_(const Div *op, const Expr &e) final;

    IndependEntry VisitExprDefault_(const Node* op, const Expr& e) final;

private:
    std::unordered_set<const Variable*> vars_;
    bool fail;

    // Combine by add
    static inline Expr AddCombine(const Expr &a, const Expr &b) {
        if (!a.defined()) return b;
        if (!b.defined()) return a;
        return ComputeExpr<Add>(a, b);
    }
    static inline Expr SubCombine(const Expr &a, const Expr &b) {
        // Check b first in case they are both undefined
        if (!b.defined()) return a;
        if (!a.defined()) return -b;
        return ComputeExpr<Sub>(a, b);
    }
    static inline Expr MulCombine(const Expr &a, const Expr &b) {
        if (!a.defined() || !b.defined())
            return Expr();
        return ComputeExpr<Mul>(a, b);
    }

    static inline bool IsImm(const Expr &e) {
        return e.defined() && (e->is_type<IntImm>() || e->is_type<UIntImm>() || e->is_type<FloatImm>());
    }
};

bool ExprSpliter::Split(const Expr &expr, Expr &depend, Expr &free) {
    auto res = VisitExpr(expr, expr);
    if (fail) {
        return false;
    }

    if (res.depend_part.defined()) {
        depend = res.depend_part;
    }
    else {
        depend = make_zero(expr.type());
    }
    if (res.free_part.defined()) {
        free = res.free_part;
    }
    else {
        free = make_zero(expr.type());
    }
    return true;
}

IndependEntry ExprSpliter::VisitExpr_(const Add* op, const Expr& e) {
    IndependEntry a = VisitExpr(op->a, op->a);
    IndependEntry b = VisitExpr(op->b, op->b);
    if (fail) {
        return IndependEntry();
    }

    // simply add two part up.
    IndependEntry ret;
    ret.depend_part = AddCombine(a.depend_part, b.depend_part);
    ret.free_part = AddCombine(a.free_part, b.free_part);
    return ret;
}

IndependEntry ExprSpliter::VisitExpr_(const Sub *op, const Expr &e) {
    IndependEntry a = VisitExpr(op->a, op->a);
    IndependEntry b = VisitExpr(op->b, op->b);
    if (fail) {
        return IndependEntry();
    }

    IndependEntry ret;
    ret.depend_part = SubCombine(a.depend_part, b.depend_part);
    ret.free_part = SubCombine(a.free_part, b.free_part);
    return ret;
}

IndependEntry ExprSpliter::VisitExpr_(const Mul *op, const Expr &e) {
    IndependEntry a = VisitExpr(op->a, op->a);
    IndependEntry b = VisitExpr(op->b, op->b);
    if (fail) {
        return IndependEntry();
    }

    /* (d1+f1)*(d2+f2) = d1*d2+f1*f2+d1*f2+d2*f1, so d1*f2 and d2*f1 must be zero (either term be undefined) or not mixed (when d1, d2 are immediates. ) */
    const Expr &d1 = a.depend_part, &f1 = a.free_part, &d2 = b.depend_part, &f2 = b.free_part;
    if ((!d1.defined() || !f2.defined() || IsImm(d1))
        && (!d2.defined() || !f1.defined() || IsImm(d2))) {
        IndependEntry ret;
        ret.depend_part = MulCombine(d1, d2);
        ret.free_part = AddCombine(AddCombine(MulCombine(f1, f2), MulCombine(d1, f2)), MulCombine(d2, f1));
        return ret;
    }
    else {
        /* can't split. */
        fail = true;
        return IndependEntry();
    }
}

IndependEntry ExprSpliter::VisitExpr_(const Variable *op, const Expr &e) {
    if (vars_.count(op) != 0) {
        IndependEntry ret;
        ret.depend_part = e;
        return ret;
    }
    else {
        IndependEntry ret;
        ret.free_part = e;
        return ret;
    }
}

class OnlyUseVarVisitor : public IRVisitor {
public:
    OnlyUseVarVisitor(const std::unordered_set<const Variable*> &vars) :
        vars_(vars), only_use(true)
    {}

    void Visit(const NodeRef& e) final {
        if (!only_use) return;
        IRVisitor::Visit(e);
    }

    void Visit_(const Variable* op) final {
        Handle(op);
    }

    void Visit_(const Load* op) final {
        Handle(op->buffer_var.get());
        IRVisitor::Visit_(op);
    }

    inline bool IsOnlyUse() const {
        return only_use;
    }

private:
    const std::unordered_set<const Variable*> &vars_;
    bool only_use;

    void Handle(const Variable *var) {
        /* if VAR is not in vars_, we can stop. */
        if (!vars_.count(var)) {
            only_use = false;
        }
    }
};

bool OnlyUseVars(const Expr &e, const std::unordered_set<const Variable*> &vars) {
    OnlyUseVarVisitor visitor(vars);
    visitor.Visit(e);
    return visitor.IsOnlyUse();
}

IndependEntry ExprSpliter::VisitExprDefault_(const Node* op, const Expr& e) {
    if (OnlyUseVars(e, vars_)) {
        /* if the expression only use variables in VARS_, or it doesn't use any variable (namely, immediate value) */
        IndependEntry ret;
        ret.depend_part = e;
        return ret;
    }
    else if (!ExprUseVar(e, vars_)) {
        /* if the expression don't use any variable in VARS_, it is the free part. */
        IndependEntry ret;
        ret.free_part = e;
        return ret;
    }
    else {
        fail = true;
        return IndependEntry();
    }
}

bool SplitExpr(const Expr &expr, const std::unordered_set<const Variable*> &vars, Expr &depend, Expr &free) {
    ExprSpliter spliter(vars);
    return spliter.Split(expr, depend, free);
}

} // namespace arith
} // namespace tvm