#include <memory>
#include <iostream>
#include <nnpusim/insn_mem.h>
#include <nnpusim/insn_decoder.h>
#include <nnpusim/controller.h>
#include <nnpusim/reg_file_module.h>
#include <nnpusim/alu.h>
#include <nnpusim/branch_unit.h>
#include <vector>

using namespace nnpu;
using namespace std;

std::vector<NNPUInsn> init_insn()
{
    vector<NNPUInsn> insns;
    using Li = nnpu::LiInsn;
    insns.emplace_back(Li(0, 5));
    insns.emplace_back(Li(1, -1));
    using Bin = nnpu::ALUBinaryInsn;
    insns.emplace_back(nnpu::BEZInsn(4, 0));
    insns.emplace_back(Bin(2, 0, 2, ALUBinaryOp::Add));
    insns.emplace_back(Bin(0, 0, 1, ALUBinaryOp::Add));
    insns.emplace_back(nnpu::JumpInsn(-3));
    insns.emplace_back(nnpu::JumpInsn(0));

    return insns;
}

WireData<bool> branchOut(int *i, int on)
{
    if (*i == on)
    {
        return WireData<bool>(true, true);
    }
    else
    {
        return WireData<bool>(false);
    }
}

int main(int argc, char *(argv[]))
{
    WireManager wm;
    vector<shared_ptr<SimModule>> modules;
    YAML::Node cfg = YAML::LoadFile("/home/jian/repositories/tvm/nnpu/nnpu_config.yaml");
    std::shared_ptr<InsnMemModule> IF(new InsnMemModule(wm, cfg));
    modules.push_back(IF);
    //cout << IF.get() << endl;
    IF->SetInsns(init_insn());
    IF->BindWires(wm);

    std::shared_ptr<InsnDecoder> ID(new InsnDecoder(wm, cfg));
    modules.push_back(ID);
    ID->BindWires(wm);

    std::shared_ptr<RegFileMod> regs(new RegFileMod(wm, cfg));
    modules.push_back(std::static_pointer_cast<SimModule>(regs));

    std::shared_ptr<Ctrl> ctrl(new Ctrl(wm, cfg, regs));
    modules.push_back(ctrl);
    ctrl->BindWires(wm);

    std::shared_ptr<ALU> alu(new ALU(wm, cfg));
    modules.push_back(alu);
    alu->BindWires(wm);

    std::shared_ptr<BranchUnit> branchUnit(new BranchUnit(wm, cfg));
    modules.push_back(branchUnit);
    branchUnit->BindWires(wm);
    
    int i;
    //wm.Get<bool>("branch_out")->SubscribeWriter(std::bind(branchOut, &i, 12));
    for (i = 0; i < 50; ++i)
    {
        cout << "end of cycle :" << i << endl;
        for (auto m : modules)
        {
            m->Move();
        }

        cout << "start of cycle :" << i + 1 << endl;

        for (auto m : modules)
        {
            m->Update();
        }
    }

    return 0;
}