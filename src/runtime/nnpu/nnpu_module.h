#ifndef TVM_RUNTIME_NNPU_NNPU_MODULE_H
#define TVM_RUNTIME_NNPU_NNPU_MODULE_H

#include <tvm/runtime/module.h>
#include <memory>
#include <vector>
#include <string>
#include "../meta_data.h"

namespace tvm
{
namespace runtime
{

/*!
 * \brief create a cuda module from generated asm code.
 *
 * \param asm_: generated asm code.
 * \param fmap: The map function information map of each function.
 * \param micro_kernels: micro kernels in string representation.
 * \param ll: Optional, printed llvm::module
 * \param ir: Optional, printed TVM IR
 */
Module NNPUModuleCreate(
    std::string asm_,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string micro_kernels,
    std::string ll,
    std::string ir);

} // end namespace runtime
}  // end namespace tvm

#endif