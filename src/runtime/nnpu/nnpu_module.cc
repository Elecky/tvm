#include "nnpu_module.h"

#include <tvm/runtime/registry.h>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include "../pack_args.h"

namespace tvm
{
namespace runtime
{

using std::string;
using std::move;

class NNPUModule : public runtime::ModuleNode
{
public:
    NNPUModule(string asm_, 
               std::unordered_map<string, tvm::runtime::FunctionInfo> fmap,
               string ll) :
        asm_code(move(asm_)),
        fmap_(move(fmap)),
        ll_(move(ll))
    {}

    PackedFunc GetFunction(
        const std::string& name,
        const std::shared_ptr<ModuleNode>& sptr_to_self) final;

private:
    string asm_code;
    std::unordered_map<string, tvm::runtime::FunctionInfo> fmap_;
    string ll_;
};

PackedFunc NNPUModule::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self)
{
    CHECK(false) << "NNPUModule::GetFunction not implemented";
    return PackedFunc();
}

Module NNPUModuleCreate(
    std::string asm_,
    std::unordered_map<std::string, tvm::runtime::FunctionInfo> fmap,
    std::string ll)
{
    auto modNode = std::make_shared<NNPUModule>(asm_, fmap, ll);
    return Module(modNode);
}

}// end namespace runtime
}  // end namespace tvm