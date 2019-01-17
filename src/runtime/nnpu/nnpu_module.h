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
 * \param asm_ generated asm code.
 * \param fmap The map function information map of each function.
 * \param ll Optional, printed llvm::module
 */
Module NNPUModuleCreate(
    std::string asm_,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string ll);

} // end namespace runtime
}  // end namespace tvm

#endif