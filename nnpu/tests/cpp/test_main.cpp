#include <memory>
#include <iostream>
#include <nnpusim/insn_mem.h>
#include <nnpusim/insn_decoder.h>
#include <nnpusim/controller.h>
#include <nnpusim/reg_file_module.h>
#include <vector>

using namespace nnpu;
using namespace std;

std::vector<NNPUInsn> init_insn()
{
    vector<NNPUInsn> insns;
    using Li = nnpu::LiInsn;
    insns.emplace_back(Li(0, 23));
    insns.emplace_back(Li(1, 0));
    insns.emplace_back(Li(2, 233));
    insns.emplace_back(Li(31, 0xff));
    insns.emplace_back(nnpu::JumpInsn(-4));

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

    std::shared_ptr<Ctrl> ctrl(new Ctrl(wm, cfg));
    modules.push_back(ctrl);
    ctrl->BindWires(wm);

    std::shared_ptr<RegFileMod> regs(new RegFileMod(wm, cfg));
    modules.push_back(std::static_pointer_cast<SimModule>(regs));
    
    int i;
    wm.Get<bool>("branch_out")->SubscribeWriter(std::bind(branchOut, &i, 12));
    for (i = 0; i < 20; ++i)
    {
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