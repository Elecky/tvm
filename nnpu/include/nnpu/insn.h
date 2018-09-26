/*
the instruction set definitions of nnpu simulator
*/

#ifndef NNPU_INSN_H
#define NNPU_INSN_H

#include <cstdint>

namespace nnpu
{

enum class InsnType
{
    VctrUnary, DMACopy, BufferLS, Li
};

struct VctrUnaryInsn
{
public:
    uint32_t VctrOutAddrReg;
    uint32_t VctrInAddrReg;
    uint32_t ElemCountReg;
};

struct DMACopyInsn
{
public:
    enum class DIR { HtoD, DtoH };

    // the copy direction
    DIR Dir;

    uint32_t HostPhyAddrReg;
    uint32_t DramAddrReg;
    uint32_t SizeReg;
};

struct BufferLSInsn
{
public:

    enum class DIR { Load, Store };

    // the copy direction
    DIR Dir;

    uint32_t DramAddrReg;
    uint32_t BufAddrReg;
    uint32_t SizeReg;
};

/*
*  Load Immediate Insn
*/
struct LiInsn
{
public:
    LiInsn() = default;

    LiInsn(uint32_t _resReg, uint32_t _imm) : ResReg(_resReg), Imm(_imm)
    {}

    uint32_t ResReg;
    uint32_t Imm;
};

struct NNPUInsn {
public:
    NNPUInsn(InsnType type) : Type(type) {}

    template <typename ... Ts>
    NNPUInsn(InsnType _type, Ts&& ... args) : Type(_type)
    {
        switch (_type)
        {
        case InsnType::Li:
            Li = LiInsn(args...);
            break;
        }
    }

    const InsnType Type;

    union {
        DMACopyInsn DMACopy;

        BufferLSInsn BufferLS;

        VctrUnaryInsn VctrUnary;

        LiInsn Li;
    };

    // a lot of constructors starts from here

    NNPUInsn(const DMACopyInsn& _insn) : Type(InsnType::DMACopy), DMACopy(_insn)
    {}

    NNPUInsn(const BufferLSInsn& _insn) : Type(InsnType::BufferLS), BufferLS(_insn)
    {}

    NNPUInsn(const VctrUnaryInsn& _insn) : Type(InsnType::VctrUnary), VctrUnary(_insn)
    {}

    NNPUInsn(const LiInsn& _insn) : Type(InsnType::Li), Li(_insn)
    {}
};

}

#endif