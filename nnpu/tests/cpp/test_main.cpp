#include <memory>
#include <iostream>
#include <nnpusim/insn_mem.h>
#include <nnpusim/insn_decoder.h>
#include <nnpusim/controller.h>
#include <nnpusim/reg_file_module.h>
#include <nnpusim/alu.h>
#include <nnpusim/branch_unit.h>
#include <nnpusim/load_store_unit.h>
#include <nnpusim/sclr_buffer.h>
#include <vector>

using namespace nnpu;
using namespace std;

std::vector<NNPUInsn> init_insn()
{
    vector<NNPUInsn> insns;
    using Li = nnpu::LiInsn;
    using Bin = nnpu::ALUBinaryInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;
    
    insns.emplace_back(Li(0, 5));
    insns.emplace_back(Li(1, -1));
    insns.emplace_back(nnpu::BEZInsn(4, 0));
    insns.emplace_back(Bin(2, 0, 2, ALUBinaryOp::Add));
    insns.emplace_back(Bin(0, 0, 1, ALUBinaryOp::Add));
    insns.emplace_back(nnpu::JumpInsn(-3));
    insns.emplace_back(nnpu::JumpInsn(0));

    InsnDumper dumper;
    for (auto &item : insns)
    {
        item.Call(dumper, cout);
        cout << endl;
    }

    return insns;
}

std::vector<NNPUInsn> load_store_test_insns()
{
    std::vector<NNPUInsn> init_insn();

    vector<NNPUInsn> insns;
    using Li = nnpu::LiInsn;
    using Bin = nnpu::ALUBinaryInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;
    using Unary = nnpu::ALURegImmInsn;

    insns.emplace_back(Li(0, 0));
    insns.emplace_back(Li(16, 7));
    insns.emplace_back(Li(1, 1));
    insns.emplace_back(Store(1, 0, 0));
    insns.emplace_back(Li(1, 2));
    insns.emplace_back(Store(1, 0, 4));
    insns.emplace_back(Li(1, 3));
    insns.emplace_back(Store(1, 0, 8));
    insns.emplace_back(Li(1, 65537));
    insns.emplace_back(Store(1, 0, 12));
    
    insns.emplace_back(Li(0, 4));  // $0 == i(4)
    //insns.emplace_back(Li(4, 4));  // $4 == 4
    //insns.emplace_back(nnpu::BEZInsn(7, 0));
    
    insns.emplace_back(Unary(2, 0, 4, ALURegImmOp::MulIU));  // $2 <- 4*i
    insns.emplace_back(Load(3, 2, -4));  // $3 <- load $2 - 4
    insns.emplace_back(Bin(3, 16, 3, ALUBinaryOp::Add));
    insns.emplace_back(Unary(0, 0, -1, ALURegImmOp::AddIU));  // i = i - 1
    insns.emplace_back(Store(3, 2, 12));

    insns.emplace_back(nnpu::BNEZInsn(-5, 0));
    //insns.emplace_back(nnpu::JumpInsn(-6));
    insns.emplace_back(Load(31, 0, 28));
    insns.emplace_back(nnpu::JumpInsn(0));

    InsnDumper dumper;
    for (auto &item : insns)
    {
        item.Call(dumper, cout);
        cout << endl;
    }

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
    IF->SetInsns(load_store_test_insns());
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

    std::shared_ptr<LoadStoreUnit> LSU(new LoadStoreUnit(wm, cfg));
    LSU->BindWires(wm);
    modules.push_back(LSU);

    std::shared_ptr<SclrBuffer> sclrBuffer(new SclrBuffer(wm, cfg));
    sclrBuffer->BindWires(wm);
    modules.push_back(sclrBuffer);
    
    int i;
    //wm.Get<bool>("branch_out")->SubscribeWriter(std::bind(branchOut, &i, 12));
    cout << "\n\n";
    for (i = 0; i < 90; ++i)
    {
        //cout << "end of cycle :" << i << endl;
        for (auto m : modules)
        {
            m->Move();
        }

        cout << "start of cycle :" << i + 1 << endl;

        for (auto m : modules)
        {
            m->Update();
        }
        /*
        for (auto m : modules)
        {
            m->Dump(DumpLevel::Brief, cout);
            //cout << endl;
        }*/
    }

    return 0;
}