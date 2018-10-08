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
        return ModeCode::W;

    case 2:
        return ModeCode::Dec;

    case 3:
        return ModeCode::Inc;

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

}  // namespace nnpu