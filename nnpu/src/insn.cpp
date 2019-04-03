/*!
* implementions of nnpu instruction member methods.
*/
#include <nnpu/insn.h>
#include <string>
#include <stdexcept>

using std::ostream;
using std::string;

namespace nnpu
{

const char* mode2str(ModeCode mode)
{
    switch (mode)
    {
    case ModeCode::N:
        return "n";
    
    case ModeCode::W:
        return "w";

    case ModeCode::Dec:
        return "dec";

    case ModeCode::Inc:
        return "inc";

    default:
        return "??";
    }
}

ModeCode ModeFromInt(uint32_t mode)
{
    switch (mode)
    {
    case 0:
        return ModeCode::N;
    
    case 1:
        return ModeCode::Inc;

    case 2:
        return ModeCode::Dec;

    case 3:
        return ModeCode::W;

    default:
        throw std::invalid_argument("unexpected mode number");
    }
}

void VctrUnaryInsn::Dump(ostream& os) const
{
    os << "V" << ToString(this->Op) << '_' << Size << "." << mode2str(this->Mode) 
       << " $" << this->VctrOutAddrReg 
       << ", $" << this->VctrInAddrReg;
}
void VReduceKeyInsn::Dump(ostream& os) const
{
    os<<"VReduceKey"<<", $"<<this->Out1AddrReg<<", $"<<this->Out2AddrReg<<", $"<<this->InAddrReg;
}
void DMACopyInsn::Dump(ostream& os) const
{
    os << "DMACopy" << ToString(this->Dir) << " $" << this->HostPhyAddrReg << ", $"
       << this->HostOffsetReg << ", $" << this->DramAddrReg << ", $" << this->SizeReg;
}

void BufferLSInsn::Dump(ostream& os) const
{
    os << ToString(this->Dir) << ".b $" << this->DramAddrReg << ", $" << this->BufAddrReg
       << ", $" << this->SizeReg;
}

void LiInsn::Dump(std::ostream& os) const
{
    os << "li $" << this->ResReg << ", " << this->Imm;
}

void StallInsn::Dump(std::ostream& os) const
{
    os << "stall";
}

void GemmInsn::Dump(std::ostream& os) const
{
    const char * insnName;
    if (ToAccBuf)
    {
        insnName = DoAcc ? "gemm.acc.up" : "gemm.acc";
    }
    else
    {
        insnName = "gemm.buf";
    }
    os << insnName << "_" << NRowOut << '_' << Factor << '_' << NColOut << '.' 
       << mode2str(this->Mode)
       << " $" << OutAddrReg << ", $" << OutRowStrideReg 
       << ", $" << In1AddrReg << ", $" << In1RowStrideReg 
       << ", $" << In2AddrReg << ", $" << In2RowStrideReg;
}

void VctrBinaryInsn::Dump(ostream& os) const
{
    os << 'V' << ToString(Op) << "V_" << Size << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << In1AddrReg << ", $" << In2AddrReg;
}

void VctrImmInsn::Dump(ostream& os) const
{
    string s= ToString(Op);
    char v='V',i='I';
    if(s[0]=='R')
    {
        s=s.substr(1,s.size()-1);
        std::swap(i,v);
    }
    os << v << s << i <<"_" << Size << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << InAddrReg << ", (IMM)"<< Imm;
}

void MatImmInsn::Dump(ostream& os) const
{
    string s(ToString(Op));
    char m='M',i='I';
    if(s[0]=='R')
    {
        s=s.substr(1,s.size()-1);
        std::swap(i,m);
    }
    os << m << s << i <<"_" << nRow << "_" << nCol << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << InAddrReg << ", (IMM)"<< Imm;
}

void VctrDotProdInsn::Dump(ostream &os) const
{
    os << "VDotV_" << Size << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << In1AddrReg << ", $" << In2AddrReg;
}

void VctrReduceInsn::Dump(std::ostream &os) const
{
    os << "VReduce" << ToString(Op) << "_" << Size << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << InAddrReg;
}

void MatBinaryInsn::Dump(std::ostream &os) const
{
    os << "M" << ToString(Op) << "M" << "_" << NRow << "_" << NCol << '.' << mode2str(Mode) 
       << " $" << OutAddrReg << ", $" << OutRowStrideReg 
       << ", $" << In1AddrReg << ", $" << In1RowStrideReg 
       << ", $" << In2AddrReg << ", $" << In2RowStrideReg;
}

void MatReduceRowInsn::Dump(std::ostream &os) const
{
    const char * insnPostfix;
    if (ToAccBuf)
    {
        insnPostfix = DoAcc ? ".acc.up" : ".acc";
    }
    else
    {
        insnPostfix = ".buf";
    }
    os << "MReduce" << ToString(Op) << "Row" << insnPostfix << '_' 
       << NRow << '_' << NCol << '.' << mode2str(Mode) 
       << " $" << OutAddrReg << ", $" << InAddrReg << ", $" << InRowStrideReg;
}

void MatReduceColInsn::Dump(std::ostream &os) const
{
    os << "MReduce" << ToString(Op) << "Col_" << NRow << '_' << NCol << '.' << mode2str(Mode) 
       << " $" << OutAddrReg << ", $" << InAddrReg;
}

void MatVctrInsn::Dump(std::ostream &os) const
{
    os << "M" << ToString(Op) << "V_" << NRow << '_' << NCol << '.' << mode2str(Mode) 
       << " $" << OutAddrReg << ", $" << OutRowStrideReg
       << ", $" << MatAddrReg << ", $" << MatRowStrideReg 
       << ", $" << VctrAddrReg;
}

void MatRowDotInsn::Dump(std::ostream &os) const
{
    const char * insn_name;
    if (toAccBuf)
    {
        insn_name = doAcc ? "MRowDot.acc.up" : "MRowDot";
    }
    else
    {
        insn_name = "MRowDot.buf";
    }
    os << insn_name << "_" << NRow << '_' << NCol << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << In1AddrReg << ", $" << In1RowStrideReg
       << ", $" << In2AddrReg << ", $" << In2RowStrideReg;
}

void VctrSclrInsn::Dump(std::ostream& os) const
{
    string s(ToString(Op));
    char m='V',i='S';
    if(s[0]=='R')
    {
        s=s.substr(1,s.size()-1);
        std::swap(i,m);
    }
    os << m << s << i <<"_" << Size << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << VctrAddrReg << ", $" << SclrAddrReg;
}

void BufferCopyInsn::Dump(ostream &os) const
{
    os << "Copy.b_" << UnitBytes << " $" << DstAddrReg << ", $" << DstStrideReg
       << ", $" << SrcAddrReg << ", $" << SrcStrideReg << ", $" << NUnitReg;
}

void CopyAcc2BufInsn::Dump(ostream &os) const
{
    os << "Copy.acc2buf_" << mode2str(Mode) << " $" << DstAddrReg
       << ", $" << SrcAddrReg << ", $" << SizeReg;
}

void MemsetInsn::Dump(ostream &os) const
{
    os << "Memset." << mode2str(Mode) << " $" << AddrReg << ", $" << NUnitReg
       << ", $" << StrideReg << ", #" << Imm;
}

void AccMemsetInsn::Dump(ostream &os) const
{
    os << "AccMemset_" << NRow << '_' << NCol << '.' << mode2str(Mode) 
       << " $" << AddrReg << ", $" << RowStrideReg << ", #" << Imm;
}

void NOPInsn::Dump(ostream &os) const
{
    os << "NOP";
}

void JumpInsn::Dump(ostream &os) const
{
    os << "Jump #" << static_cast<int32_t>(Offset);
}

void BEZInsn::Dump(ostream &os) const
{
    os << "BEZ $" << CondReg << ", #" << static_cast<int32_t>(Offset);
}

void BNEZInsn::Dump(ostream &os) const
{
    os << "BNEZ $" << CondReg << ", #" << static_cast<int32_t>(Offset);
}

void ALUBinaryInsn::Dump(ostream &os) const
{
    os << ToString(Op) << " $" << RdReg << ", $" << RsReg << ", $" << RtReg;
}

void SclrLoadInsn::Dump(ostream &os) const
{
    os << "Load.S $" << RdReg << ", ($" << AddrReg << " + " << Offset << ")";
}

void SclrStoreInsn::Dump(ostream &os) const
{
    os << "Store.S $" << RsReg << ", ($" << AddrReg << " + " << Offset << ")";
}

void ALURegImmInsn::Dump(ostream &os) const
{
    os << ToString(Op) << " $" << RdReg << ", $" << RsReg << ", #" << Imm;
}

void DMA2BufferInsn::Dump(ostream& os) const
{
    os << "DMA2Buffer" << ToString(this->Dir) << " $" << this->HostPhyAddrReg << ", $"
       << this->HostOffsetReg << ", $" << this->BufAddrReg << ", $" << this->SizeReg;
}

// ToString functions starts from here

const char* ToString(ALUBinaryOp op)
{
    switch (op)
    {
    case ALUBinaryOp::Add:
        return "AddU";
    case ALUBinaryOp::Sub:
        return "SubU";
    case ALUBinaryOp::Mul:
        return "MulU";
    case ALUBinaryOp::DivU:
        return "DivU";
    case ALUBinaryOp::ModU:
        return "ModU";
    case ALUBinaryOp::SLTU:
        return "SLTU";
    case ALUBinaryOp::SLT:
        return "SLT";
    case ALUBinaryOp::SEQ:
        return "SEQ";
    case ALUBinaryOp::XOR:
        return "XOR";
    case ALUBinaryOp::And:
        return "And";
    case ALUBinaryOp::Or:
        return "Or";
    
    default:
        return "Unknown";
    }
}

const char * ToString(ALURegImmOp op)
{
    switch (op)
    {
    case ALURegImmOp::AddIU:
        return "AddIU";
    case ALURegImmOp::ISubU:
        return "ISubU";
    case ALURegImmOp::MulIU:
        return "MulIU";
    case ALURegImmOp::DivIU:
        return "DivIU";
    case ALURegImmOp::ModIU:
        return "ModIU";
    case ALURegImmOp::SLTIU:
        return "SLTIU";
    case ALURegImmOp::SLTI:
        return "SLTI";
    case ALURegImmOp::SEQI:
        return "SEQI";
    case ALURegImmOp::XORI:
        return "XORI";
    case ALURegImmOp::AndI:
        return "AndI";
    case ALURegImmOp::OrI:
        return "OrI";
    case ALURegImmOp::SHLI:
        return "SHLI";

    default:
        return "UnkownRegImm";
    }
}

const char* ToString(VctrUnaryOp value)
{
    switch (value)
    {
    case VctrUnaryOp::Exp:
        return "Exp";
    case VctrUnaryOp::Log:
        return "Log";
    default:
        return "Unhandled";
    }
}

const char* ToString(DMADIR value)
{
    switch (value)
    {
    case DMADIR::DtoH:
        return "DtoH";

    case DMADIR::HtoD:
        return "HtoD";

    default:
        return "Unknown";
    }
}

const char* ToString(LSDIR value)
{
    switch (value)
    {
    case LSDIR::Load:
        return "Load";

    case LSDIR::Store:
        return "Store";

    default:
        return "Unknown";
    }
}

const char* ToString(VctrBinaryOp value)
{
    switch (value)
    {
    case VctrBinaryOp::Add:
        return "Add";

    case VctrBinaryOp::Sub:
        return "Sub";

    case VctrBinaryOp::Mul:
        return "Mul";

    case VctrBinaryOp::Div:
        return "Div";

    case VctrBinaryOp::GTM:
        return "GTM";

    default:
        return "Unknown";
    }
}

const char* ToString(ReduceOp value)
{
    switch (value)
    {
    case ReduceOp::Sum:
        return "Sum";
    
    case ReduceOp::Max:
        return "Max";

    case ReduceOp::Min:
        return "Min";

    default:
        return "Unknown";
    }
}
const char* ToString(VctrImmOp value)
{
    switch (value)
    {
    case VctrImmOp::Add:
        return "Add";

    case VctrImmOp::Sub:
        return "Sub";

    case VctrImmOp::Mul:
        return "Mul";

    case VctrImmOp::Div:
        return "Div";

    case VctrImmOp::GTM:
        return "GTM";
    
    case VctrImmOp::RSub:
        return "RSub";

    case VctrImmOp::RDiv:
        return "RDiv";
    
    default:
        return "Unknown";
    }
}

const char* ToString(VctrSclrOp value)
{
    switch (value)
    {
    case VctrSclrOp::Add:
        return "Add";

    case VctrSclrOp::Sub:
        return "Sub";

    case VctrSclrOp::Mul:
        return "Mul";

    case VctrSclrOp::Div:
        return "Div";

    case VctrSclrOp::GTM:
        return "GTM";
    
    case VctrSclrOp::RSub:
        return "RSub";

    case VctrSclrOp::RDiv:
        return "RDiv";
    
    default:
        return "Unknown";
    }
}


const char* ToString(MatImmOp value)
{
    switch (value)
    {
    case MatImmOp::Add:
        return "Add";

    case MatImmOp::Mul:
        return "Mul";
    
    case MatImmOp::RSub:
        return "RSub";

    default:
        return "Unknown";
    }
}

const char* ToString(MatBinaryOp value)
{
    switch (value)
    {
    case MatBinaryOp::Add:
        return "Add";
    
    case MatBinaryOp::Sub:
        return "Sub";
    
    case MatBinaryOp::Mul:
        return "Mul";
    
    default:
        return "Unknown";
    }
}

const char* ToString(MatVctrOp value)
{
    switch (value)
    {
    case MatVctrOp::Add:
        return "Add";
    
    case MatVctrOp::Sub:
        return "Sub";
    
    case MatVctrOp::Mul:
        return "Mul";
    
    default:
        return "Unknown";
    }
}

KVList_t ALUBinaryInsn::GetRegMap() const
{
    KVList_t res;
    res[RsReg] = 0;
    res[RtReg] = 0;

    return res;
}

KVList_t VctrUnaryInsn::GetRegMap() const
{
    KVList_t res;
    res[VctrOutAddrReg] = 0;
    res[VctrInAddrReg] = 0;

    return res;
}

KVList_t DMACopyInsn::GetRegMap() const
{
    KVList_t res;
    res[HostPhyAddrReg] = 0;
    res[HostOffsetReg] = 0;
    res[DramAddrReg] = 0;
    res[SizeReg] = 0;

    return res;
}

KVList_t BufferLSInsn::GetRegMap() const
{
    KVList_t res;
    res[DramAddrReg] = 0;
    res[BufAddrReg] = 0;
    res[SizeReg] = 0;

    return res;
}

KVList_t BufferCopyInsn::GetRegMap() const
{
    KVList_t res;
    res[DstAddrReg] = 0;
    res[DstStrideReg] = 0;
    res[SrcAddrReg] = 0;
    res[SrcStrideReg] = 0;
    res[NUnitReg] = 0;

    return res;
}

KVList_t MemsetInsn::GetRegMap() const
{
    KVList_t res;
    res[AddrReg] = 0;
    res[NUnitReg] = 0;
    res[StrideReg] = 0;

    return res;
}

KVList_t AccMemsetInsn::GetRegMap() const
{
    KVList_t res;
    res[AddrReg] = 0;
    res[RowStrideReg] = 0;

    return res;
}

KVList_t LiInsn::GetRegMap() const
{
    return KVList_t();  // Li insn has no register operand.
}

KVList_t GemmInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[OutRowStrideReg] = 0;
    res[In1AddrReg] = 0;
    res[In1RowStrideReg] = 0;
    res[In2AddrReg] = 0;
    res[In2RowStrideReg] = 0;

    return res;
}

KVList_t VReduceKeyInsn::GetRegMap() const
{
    KVList_t res;
    res[Out1AddrReg] = 0;
    res[Out2AddrReg] = 0;
    res[InAddrReg] = 0;
    return res;
}

KVList_t VctrBinaryInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[In1AddrReg] = 0;
    res[In2AddrReg] = 0;

    return res;
}

KVList_t VctrImmInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[InAddrReg] = 0;

    return res;
}

KVList_t MatImmInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[InAddrReg] = 0;
    
    return res;
}

KVList_t VctrDotProdInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[In1AddrReg] = 0;
    res[In2AddrReg] = 0;

    return res;
}

KVList_t VctrReduceInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[InAddrReg] = 0;

    return res;
}

KVList_t MatBinaryInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[OutRowStrideReg] = 0;
    res[In1AddrReg] = 0;
    res[In1RowStrideReg] = 0;
    res[In2AddrReg] = 0;
    res[In2RowStrideReg] = 0;

    return res;
}

KVList_t MatReduceRowInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[InAddrReg] = 0;
    res[InRowStrideReg] = 0;

    return res;
}

KVList_t MatReduceColInsn::GetRegMap() const
{
    throw std::logic_error("Not implemented");
}

KVList_t MatVctrInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[OutRowStrideReg] = 0;
    res[MatAddrReg] = 0;
    res[MatRowStrideReg] = 0;
    res[VctrAddrReg] = 0;

    return res;
}

KVList_t MatRowDotInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[In1AddrReg] = 0;
    res[In1RowStrideReg] = 0;
    res[In2AddrReg] = 0;
    res[In2RowStrideReg] = 0;

    return res;
}

KVList_t VctrSclrInsn::GetRegMap() const
{
    KVList_t res;
    res[OutAddrReg] = 0;
    res[VctrAddrReg] = 0;
    res[SclrAddrReg] = 0;

    return res;
}

KVList_t CopyAcc2BufInsn::GetRegMap() const
{
    KVList_t res;
    res[DstAddrReg] = 0;
    res[SrcAddrReg] = 0;
    res[SizeReg] = 0;

    return res;
}

KVList_t NOPInsn::GetRegMap() const
{
    return KVList_t();
}

KVList_t JumpInsn::GetRegMap() const
{
    return KVList_t();
}

KVList_t BEZInsn::GetRegMap() const
{
    KVList_t res;
    res[CondReg] = 0;
    
    return res;
}

KVList_t BNEZInsn::GetRegMap() const
{
    KVList_t res;
    res[CondReg] = 0;
    
    return res;
}

KVList_t StallInsn::GetRegMap() const
{
    return KVList_t();
}

KVList_t SclrLoadInsn::GetRegMap() const
{
    KVList_t res;
    res[AddrReg] = 0;

    return res;
}

KVList_t SclrStoreInsn::GetRegMap() const
{
    KVList_t res;
    res[AddrReg] = 0;
    res[RsReg] = 0;

    return res;
}

KVList_t ALURegImmInsn::GetRegMap() const
{
    KVList_t res;
    res[RsReg] = 0;

    return res;
}

KVList_t DMA2BufferInsn::GetRegMap() const
{
    KVList_t res;
    res[HostPhyAddrReg] = 0;
    res[HostOffsetReg] = 0;
    res[BufAddrReg] = 0;
    res[SizeReg] = 0;
    return res;
}

}  // namespace nnpu