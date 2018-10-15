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
    os << "gemm_" << NRowOut << '_' << Factor << '_' << NColOut << '.' << mode2str(this->Mode)
       << " $" << OutAddrReg << ", $" << In1AddrReg << ", $" << In2AddrReg;
}

void VctrBinaryInsn::Dump(ostream& os) const
{
    os << 'V' << ToString(Op) << "V_" << Size << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << In1AddrReg << ", $" << In2AddrReg;
}

// some ploblem
void VctrImmInsn::Dump(ostream& os) const
{
    os << 'V' << ToString(Op) << "I_" << Size << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << InAddrReg << ", (IMM)"<< Imm;
}

void MatImmInsn::Dump(ostream& os) const
{
    os << 'M' << ToString(Op) << "I_" << nRow << "_" << nCol << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << InAddrReg << ", (IMM)"<< Imm;
}

void VctrDotProdInsn::Dump(ostream &os) const
{
    os << "VDotV_" << Size << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << In1AddrReg << ", $" << In2AddrReg;
}

void VctrReduceInsn::Dump(std::ostream& os) const
{
    os << "VReduce" << ToString(Op) << "_" << '.' << mode2str(Mode) << " $"
       << OutAddrReg << ", $" << InAddrReg;
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

}  // namespace nnpu