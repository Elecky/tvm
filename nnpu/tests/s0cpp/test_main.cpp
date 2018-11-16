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
    insn.push_back(Li(0, 10));
    insn.push_back(Li(1, 0));
    insn.push_back(Bin(1, 1, 0, ALUBinaryOp::Add));
    insn.push_back(Unary(0, 0, -1, ALURegImmOp::AddIU));
    insn.push_back(nnpu::BNEZInsn(-2, 0));
    return insn;
}
std::vector<NNPUInsn> sum100plus(){
    std::vector<NNPUInsn> insn;
    using Li = nnpu::LiInsn;
    using Bin = nnpu::ALUBinaryInsn;
    using Unary = nnpu::ALURegImmInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;
    insn.push_back(Li(0, 100));
    insn.push_back(Li(1, 0));
    insn.push_back(Bin(1, 1, 0, ALUBinaryOp::Add));
    insn.push_back(Store(1,0,8));
    insn.push_back(Load(1,0,8));
    insn.push_back(Unary(0, 0, -1, ALURegImmOp::AddIU));
    insn.push_back(nnpu::BNEZInsn(-4, 0));
    return insn;
}
std::vector<NNPUInsn> insert_sort_insns()
{
    vector<NNPUInsn> insns;
    using Li = nnpu::LiInsn;
    using Bin = nnpu::ALUBinaryInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;
    using Unary = nnpu::ALURegImmInsn;

    insns.emplace_back(Li(0, 0));
    insns.emplace_back(Li(1, 29));
    insns.emplace_back(Store(1, 0, 0));
    insns.emplace_back(Li(1, 255));
    insns.emplace_back(Store(1, 0, 4));
    insns.emplace_back(Li(1, 8));
    insns.emplace_back(Store(1, 0, 8));
    insns.emplace_back(Li(1, 65537));
    insns.emplace_back(Store(1, 0, 12));
    insns.emplace_back(Li(1, 233));
    insns.emplace_back(Store(1, 0, 16));

    // insert sort
    insns.emplace_back(Li(1, 1));  // let $1=i;  $1 <= 1

    insns.emplace_back(Unary(2, 1, 5, ALURegImmOp::SLTIU));  // $2 = $1 < 5
    insns.emplace_back(BEZInsn(13, 2));  // BEZ $2, #??

    insns.emplace_back(Unary(3, 1, 4, ALURegImmOp::MulIU));  // let $3=j <= 4 * i
    insns.emplace_back(Load(4, 3, 0));  // let $4 = arr[i]

    insns.emplace_back(BEZInsn(7, 3));  // if j == 0, end while
    insns.emplace_back(Load(5, 3, -4));  // load a[j / 4 - 1]
    insns.emplace_back(Bin(2, 4, 5, ALUBinaryOp::SLTU));  // key < a[j / 4 - 1] ?
    insns.emplace_back(BEZInsn(4, 2));  // if not, end while
    // if key is less
    insns.emplace_back(Store(5, 3, 0));  // a[j / 4] = a[j / 4 - 1]
    insns.emplace_back(Unary(3, 3, -4, ALURegImmOp::AddIU));  // j = j - 4
    insns.emplace_back(JumpInsn(-6));

    // end of while
    insns.emplace_back(Store(4, 3, 0));


    insns.emplace_back(Unary(1, 1, 1, ALURegImmOp::AddIU));
    insns.emplace_back(JumpInsn(-13));

    // load all for once
    insns.emplace_back(Li(1, 0));  // let $1=i;  $1 <= 0

    insns.emplace_back(Unary(2, 1, 20, ALURegImmOp::SLTIU));  // $2 = $1 < 20
    insns.emplace_back(BEZInsn(4, 2));
    insns.emplace_back(Load(16, 1, 0));
    insns.emplace_back(Unary(1, 1, 4, ALURegImmOp::AddIU));
    insns.emplace_back(JumpInsn(-4));
    //insns.emplace_back(JumpInsn(0));
    return insns;
}
int main(int argc, char *(argv[]))
{
    YAML::Node cfg = YAML::LoadFile("nnpu_config.yaml");
    nnpu::S0Simulator myS0(cfg);
    myS0.Run(insert_sort_insns());
    cout<<myS0.ReadReg(0)<<endl;
    cout<<myS0.ReadReg(1)<<endl;
    cout<<"hello world"<<endl;
    return 0;
}