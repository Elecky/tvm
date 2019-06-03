/*!
 *  Copyright (c) 2017 by Contributors
 * \file extract_micro_code.h
 * \brief extract micro code from device function.
 */
#ifndef TVM_PASS_EXTRACT_MICRO_CODE_H
#define TVM_PASS_EXTRACT_MICRO_CODE_H

#include <tvm/ir.h>
#include <tvm/lowered_func.h>
#include <tvm/channel.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/runtime/module.h>
#include <unordered_map>

namespace tvm {
namespace ir {

class micro_code_extractor : public IRMutator {
public:
    /**!
     * \brief construct a micro code extractor. it begins micro code extraction after AttrStmt with a key ATTR_KEY was met.
     * \param uop_templates: templates of UOP instructions. TODO: definition of the format!!!
    */
    micro_code_extractor(const std::unordered_map<std::string, std::string> *uop_templates,
                         unsigned start_idx = 0,
                         std::string _attr_key = "coproc_uop_scope");

    Stmt Mutate_(const AttrStmt* op, const Stmt& s) final override;

    std::pair<Stmt, std::vector<std::string> > extract(const Stmt &body);

private:
    std::vector<std::string> micro_kernels;
    unsigned start_idx_;
    const std::unordered_map<std::string, std::string> *uop_templates_;
    std::string attr_key_;
};

std::pair<Array<LoweredFunc>, std::vector<std::string> >
generate_micro_kernels(const Array<LoweredFunc> &funcs, const std::unordered_map<std::string, std::string> &uop_templates);

}  // namespace tvm
}  // namespace ir

#endif