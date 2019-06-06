#include <tvm/ir_functor_ext.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir.h>

namespace tvm
{
namespace arith
{

bool SplitExpr(const Expr &expr, const std::unordered_set<const Variable*> &vars, Expr &depend, Expr &free);

} // namespace arith
} // namespace tvm
