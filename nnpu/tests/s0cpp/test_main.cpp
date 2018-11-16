#include <memory>
#include <iostream>
#include <nnpusim/S0Simulator.h>
#include <vector>
#include <yaml-cpp/yaml.h>
using namespace nnpu;
using namespace std;

std::vector<NNPUInsn> sum100(){
    std::vector<NNPUInsn> insn;
    using Li = nnpu::LiInsn;
    using Bin = nnpu::ALUBinaryInsn;
    using Unary = nnpu::ALURegImmInsn;
    const int datasize=4;
    insn.push_back(Li(0, 100));
    insn.push_back(Li(1, 0));
    insn.push_back(Bin(1, 1, 0, ALUBinaryOp::Add));
    insn.push_back(Unary(0, 0, -1, ALURegImmOp::AddIU));
    insn.push_back(nnpu::BNEZInsn(2, 0));
    return insn;
}
std::vector<NNPUInsn> sum100plus(){
    std::vector<NNPUInsn> insn;
    using Li = nnpu::LiInsn;
    using Bin = nnpu::ALUBinaryInsn;
    using Unary = nnpu::ALURegImmInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;
    const int datasize=4;
    insn.push_back(Li(0, 100));
    insn.push_back(Li(1, 0));
    insn.push_back(Bin(1, 1, 0, ALUBinaryOp::Add));
    insn.push_back(Store(1,0,8));
    insn.push_back(Load(1,0,8));
    insn.push_back(Unary(0, 0, -1, ALURegImmOp::AddIU));
    insn.push_back(nnpu::BNEZInsn(4, 0));
    return insn;
}

int main(int argc, char *(argv[]))
{
    YAML::Node cfg = YAML::LoadFile("nnpu_config.yaml");
    nnpu::S0Simulator myS0(cfg);
    myS0.Run(sum100plus());
    cout<<myS0.ReadReg(0)<<endl;
    cout<<myS0.ReadReg(1)<<endl;
    cout<<"hello world"<<endl;
    return 0;
}