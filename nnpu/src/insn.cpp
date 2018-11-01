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
    os << "V" << ToString(this->Op) << "." << mode2str(this->Mode) << " $" << this->VctrOutAddrReg 
       << ", $" << this->VctrInAddrReg << ", $" << this->ElemCountReg;
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
    string insnName;
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
    os << "MReduce" << ToString(Op) << "Row_" << NRow << '_' << NCol << '.' << mode2str(Mode) 
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
    os << "MRowDot" << "_" << NRow << '_' << NCol << '.' << mode2str(Mode) << " $"
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
    os << "Copy.acc2buf_" << UnitBytes << " $" << DstAddrReg << ", $" << DstStrideReg
       << ", $" << SrcAddrReg << ", $" << SrcStrideReg << ", $" << NUnitReg;
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

// ToString functions starts from here

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

}  // namespace nnpu