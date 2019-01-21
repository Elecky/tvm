/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_nnpu.cc
 * \brief NNPU code generator.
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/device_api.h>
#include "codegen_llvm.h"
#include "../build_common.h"
#include "../../pass/ir_util.h"
#include "../../runtime/nnpu/nnpu_module.h"

namespace tvm
{
namespace codegen
{

// NNPU code generator.
class CodeGenNNPU : public CodeGenLLVM
{
public:
  /*!
   * \brief Compile and add NNPU function f to the current module.
   *        we have to convert arguments of handle type to i32 type arguments,
   *        actually NNPU device function can only accept buffer_var arguments,
   *        which will be converted to physical address of the buffers during
   *        code generation.
   * \param f The function to be added.
   */
  void AddFunction(const LoweredFunc &f) final;

  llvm::Value *VisitExpr_(const Call *op) final;

  // used for debugging, disable optimize
  std::unique_ptr<llvm::Module> Finish() final;

  // WARNING: CodeGenLLVM has no virtual destructor, so we shouldn't add any field member!!
  //          since it's common to save a pointer to CodeGenLLVM derivate in unique_ptr.
};

void CodeGenNNPU::AddFunction(const LoweredFunc &f)
{
  constexpr bool ret_void{true};
  this->InitFuncState();
  std::vector<llvm::Type *> arg_types;
  is_restricted_ = f->is_restricted;
  // get argument types.
  for (Var arg : f->args)
  {
    Type t = arg.type();
    if (t.is_handle())
    {
      // all handle type arguments are converted to i32 type.
      // since NNPU device function can only accept host buffer var handle type,
      // which we convert to the physical address of corresponding buffer when
      // call device function.
      arg_types.push_back(t_int32_);
    }
    else
    {
      arg_types.push_back(LLVMType(arg.type()));
    }
  }
  llvm::FunctionType *ftype = llvm::FunctionType::get(
      ret_void ? t_void_ : t_int_, arg_types, false);
  CHECK(module_->getFunction(f->name) == nullptr)
      << "Function " << f->name << " already exist in module";
  function_ = llvm::Function::Create(
      ftype, llvm::Function::ExternalLinkage,
      f->name, module_.get());
  function_->setCallingConv(llvm::CallingConv::C);
  function_->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
  // set argument var map and align information
  auto arg_it = function_->arg_begin();
  for (size_t i = 0; i < f->args.size(); ++i, ++arg_it)
  {
    llvm::Argument *v = &(*arg_it);
    const Var &var = f->args[i];
    var_map_[var.get()] = v; // mapping from TVM IR variable to LLVM variable of arguments.
    // NNPU does not need noalias attribute.
  }
  llvm::BasicBlock *entry = llvm::BasicBlock::Create(*ctx_, "entry", function_);
  builder_->SetInsertPoint(entry);
  this->VisitStmt(f->body);
  if (ret_void)
  {
    builder_->CreateRetVoid();
  }
  else
  {
    builder_->CreateRet(ConstInt32(0));
  }
}

llvm::Value *CodeGenNNPU::VisitExpr_(const Call *op)
{
  if (op->is_intrinsic("llvm_intrin_with_side_effct"))
  {
    CHECK_GE(op->args.size(), 3U);
    // the first 3 arguments are intrinsic-id, num_signature and has_return(uint)
    llvm::Intrinsic::ID id = static_cast<llvm::Intrinsic::ID>(
        op->args[0].as<UIntImm>()->value);
    uint64_t num_signature = op->args[1].as<UIntImm>()->value;
    uint64_t has_return = op->args[2].as<UIntImm>()->value;

    std::vector<llvm::Value *> arg_value;
    std::vector<llvm::Type *> sig_type;
    for (size_t i = 3; i < op->args.size(); ++i)
    {
      arg_value.push_back(MakeValue(op->args[i]));
      if (i - 3 < num_signature)
      {
        sig_type.push_back(arg_value.back()->getType());
      }
    }
    llvm::Type *return_type{nullptr};
    if (has_return)
      return_type = LLVMType(op->type);
    else
      return_type = t_void_;
    // if (sig_type.size() > 0 && return_type != sig_type[0])
    // {
    //   sig_type.insert(sig_type.begin(), return_type);
    //   LOG(INFO) << "----------we are adding return type";
    // }
    llvm::Function *f = llvm::Intrinsic::getDeclaration(
        // module_.get(), id, sig_type);
        module_.get(), id);
    return builder_->CreateCall(f, arg_value);
  }
  else
  {
    return CodeGenLLVM::VisitExpr_(op);
  }
}

std::unique_ptr<llvm::Module> CodeGenNNPU::Finish() {
  this->AddStartupFunction();
  // link modules
  for (size_t i = 0; i < link_modules_.size(); ++i) {
    CHECK(!llvm::Linker::linkModules(*module_, std::move(link_modules_[i])))
        << "Failed to link modules";
  }
  link_modules_.clear();
  // disable optimize
  return std::move(module_);
}

runtime::Module BuildNNPU(Array<LoweredFunc> funcs, std::string target)
{
  CHECK(target.length() >= 4 &&
        target.substr(0, 4) == "nnpu");
  InitializeLLVM();  // initialize LLVM, forgetting to do this, llvm even can't find target.
  std::ostringstream config;
  config << "-mtriple=NNPU";
  llvm::TargetMachine *tm = GetLLVMTargetMachine(
                              config.str(),
                              false,
                              llvm::Reloc::Static);
  std::unique_ptr<CodeGenNNPU> cg(new CodeGenNNPU());
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext());
  cg->Init(funcs[0]->name, tm, ctx.get(), false, false);
  for (LoweredFunc f : funcs)
  {
    cg->AddFunction(f);
  }

  std::unique_ptr<llvm::Module> module = cg->Finish();
  llvm::SmallString<8> data_asm, data_ll;
  llvm::raw_svector_ostream dest_asm(data_asm), dest_ll(data_ll);
  dest_asm.SetUnbuffered();
  dest_ll.SetUnbuffered();
  // print ll
  module->print(dest_ll, nullptr);
  std::string ll(data_ll.begin(), data_ll.end());
  // std::cout << "The generate LLVM IR is:\n";
  // std::cout << ll;

  // emit asm
  llvm::legacy::PassManager pass;
#if TVM_LLVM_VERSION <= 60
  CHECK(tm->addPassesToEmitFile(
            pass, dest_asm, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#else
  CHECK(tm->addPassesToEmitFile(
            pass, dest_asm, nullptr, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#endif
  pass.run(*module);
  std::string asm_code(data_asm.begin(), data_asm.end());
  // std::cout << "The generate asm is:\n";
  // std::cout << asm_code;
  return NNPUModuleCreate(asm_code, ExtractFuncInfo(funcs), ll);
}

TVM_REGISTER_API("codegen.build_nnpu")
    .set_body([](TVMArgs args, TVMRetValue *rv) {
      *rv = BuildNNPU(args[0], args[1]);
    }); 

} // namespace codegen
} // namespace tvm
#endif // TVM_LLVM_VERSION