#include <memory>
#include <iostream>
#include <nnpusim/insn_mem.h>
#include <nnpusim/insn_decoder.h>
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
    insns.emplace_back(nnpu::JumpInsn(0));

    return insns;
}

int main(int argc, char *(argv[]))
{
    WireManager wm;
    YAML::Node cfg = YAML::LoadFile("/home/jian/repositories/tvm/nnpu/nnpu_config.yaml");
    std::shared_ptr<InsnMemModule> IF(new InsnMemModule(wm, cfg));
    //cout << IF.get() << endl;
    IF->SetInsns(init_insn());
    IF->BindWires(wm);

    std::shared_ptr<InsnDecoder> ID(new InsnDecoder(wm, cfg));
    ID->BindWires(wm);

    auto IFOut = wm.Get<InsnWrapper>("ID_out");
    InsnDumper dumper;
    for (int i = 0; i < 10; ++i)
    {
        IF->Move();
        ID->Move();
        auto ifOut = IFOut->Read();

        if (ifOut.HasData)
        {
            cout << "# " << ifOut.Data.pc << ": ";
            ifOut.Data.insn->Call(dumper, cout);
            cout << endl;
        }

        IF->Update();
        ID->Update();
    }

    return 0;
}