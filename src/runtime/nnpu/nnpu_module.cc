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

    const char* type_key() const
    {
        return "NNPU";
    }

    PackedFunc GetFunction(
        const std::string& name,
        const std::shared_ptr<ModuleNode>& sptr_to_self) final;

    std::string GetSource(const std::string& format) final;

    const std::string& GetAsmCode() const
    {
        return asm_code;
    };

    std::string GetAsmCode()
    {
        return asm_code;
    };

private:
    string asm_code;
    std::unordered_map<string, tvm::runtime::FunctionInfo> fmap_;
    string ll_;
};

/*!
 * \brief 
*/
class NNPUWrappedFunc {
public:
    NNPUWrappedFunc(std::shared_ptr<ModuleNode> _sptr,
                    std::string _func_name)
        : sptr_(_sptr),
          func_name_(_func_name)
    {
    }

    void operator()(TVMArgs args, TVMRetValue* rv) const;

private:
    // the module resource holder
    std::shared_ptr<ModuleNode> sptr_;

    std::string func_name_;
};

void NNPUWrappedFunc::operator()(TVMArgs args, 
                                 TVMRetValue *rv) const 
{
    auto &os = LOG(INFO);
    os << "\n NNPUWrappedFunc called, function name = " << func_name_
       << "\n argc = " << args.num_args << ", arguments = ";
    auto handleToPhyAddr =  tvm::runtime::Registry::Get("nnpu.handleToPhyAddr");
    CHECK(handleToPhyAddr != nullptr);
    for (int i = 0; i < args.num_args - 1; ++i)
    {
        os << "\n arg" << i << " = " << (int64_t)(*handleToPhyAddr)((void*)args[i]);
    }
    os << "\n coproc_scope = " << static_cast<int>(args[5]) << std::endl;
}

PackedFunc NNPUModule::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self)
{
    CHECK_EQ(sptr_to_self.get(), this);
    CHECK_NE(name, symbol::tvm_module_main)
        << "Device function do not have main";
    auto it = fmap_.find(name);
    if (it == fmap_.end()) return PackedFunc();
    NNPUWrappedFunc f(sptr_to_self, name);
    // Note: it seems that the tvm.call_packed created for host code 
    //       has more arguments than the packed function parameter!!
    return PackedFunc(f);
}

std::string NNPUModule::GetSource(const std::string& format) {
    if (format == "asm") 
        return asm_code;
    else if (format == "ll")
        return ll_;
    else
        return "";
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