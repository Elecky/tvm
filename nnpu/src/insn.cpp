/*!
* implementions of nnpu instruction member methods.
*/
#include <nnpu/insn.h>
#include <string>

using std::ostream;
using std::string;

namespace nnpu
{

const char* mode2str(uint32_t mode)
{
    switch (mode)
    {
    case 0:
        return "n";
    
    case 1:
        return "w";

    case 2:
        return "dec";

    case 3:
        return "inc";

    default:
        return "??";
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