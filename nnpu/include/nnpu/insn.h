/*
the instruction set definitions of nnpu simulator
*/

#ifndef NNPU_INSN_H
#define NNPU_INSN_H

#include <cstdint>
#include <iostream>
#include <dmlc/logging.h>

namespace nnpu
{

/*
* instruction type, similar to opcode in MIPS instruction
*/
enum class InsnType
{
    VctrUnary, DMACopy, BufferLS, Li
};

/*!
 * Mode type for instructios, which indicates the input, output data type.
*/
enum class ModeCode { N, W, Dec, Inc };

ModeCode ModeFromInt(uint32_t mode);

/*
* unary vector instruction
*/
enum class VctrUnaryOp { Exp, Log };
const char* ToString(VctrUnaryOp value);
struct VctrUnaryInsn
{
public:
    /*!
    * \brief the unary vector instruction operation, like the Funct in MIPS insn.
    */
    VctrUnaryOp Op;
    uint32_t VctrOutAddrReg;
    uint32_t VctrInAddrReg;
    uint32_t ElemCountReg;

    ModeCode Mode;

    /* default constructor */
    VctrUnaryInsn() = default;

    /* constructor */
    VctrUnaryInsn(VctrUnaryOp _op, uint32_t _vctrOutAddrReg, uint32_t _vctrInAddrReg, 
        uint32_t _elemCountReg, ModeCode _mode) :
        Op(_op), VctrOutAddrReg(_vctrOutAddrReg), VctrInAddrReg(_vctrInAddrReg), 
        ElemCountReg(_elemCountReg), Mode(_mode)
    {}

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;
};

/*!
 * \brief DMA copy instruction.
*/
enum class DMADIR { HtoD, DtoH };
const char* ToString(DMADIR value);
struct DMACopyInsn
{
public:
    // the DMA copy direction
    DMADIR Dir;
    uint32_t HostPhyAddrReg;
    uint32_t HostOffsetReg;
    uint32_t DramAddrReg;
    /* copy size in byte
    */
    uint32_t SizeReg;

    /* default constructor */
    DMACopyInsn() = default;

    /* constructor */
    DMACopyInsn(DMADIR _dir, uint32_t _hostPhyAddrReg, uint32_t _hostOffsetReg, 
        uint32_t _dramAddrReg, uint32_t _sizeReg) :
        Dir(_dir), HostPhyAddrReg(_hostPhyAddrReg), HostOffsetReg(_hostOffsetReg), 
        DramAddrReg(_dramAddrReg), SizeReg(_sizeReg)
    {}

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;
};

/*
* Scratchpad load and store instruction.
*/
enum class LSDIR { Load, Store };
const char* ToString(LSDIR value);
struct BufferLSInsn
{
public:
    // the copy direction
    LSDIR Dir;

    uint32_t DramAddrReg;
    uint32_t BufAddrReg;
    /* copy size in byte */
    uint32_t SizeReg;

    /* default constructor */
    BufferLSInsn() = default;

    /* constructor */
    BufferLSInsn(LSDIR _dir, uint32_t _dramAddrReg, uint32_t _bufAddrReg, uint32_t _sizeReg) :
        Dir(_dir), DramAddrReg(_dramAddrReg), BufAddrReg(_bufAddrReg), SizeReg(_sizeReg)
    {}

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;
};

/*!
* \brief Load Immediate Insn,
*        assign an immediate value to a register
*/
struct LiInsn
{
public:
    uint32_t ResReg;
    uint32_t Imm;

    /* default constructor */
    LiInsn() = default;

    LiInsn(uint32_t _resReg, uint32_t _imm) : ResReg(_resReg), Imm(_imm)
    {}

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;
};

/*
* \brief nnpu instruction struct, contains a union of actual instructions, 
*        and a InsnType field.
*/
struct NNPUInsn {
public:
    NNPUInsn(InsnType type) : Type(type) {}

    const InsnType Type;

    // instruction union
    union {
        DMACopyInsn DMACopy;

        BufferLSInsn BufferLS;

        VctrUnaryInsn VctrUnary;

        LiInsn Li;
    };

    /* dispatch a call depends on the instruction type
    * params:
    *   functor: a callable object, whose first parameter will be the inner instruction, 
    *            followed by args..., and the return type is decltype(functor)::result_type.
    *   args...: other parameters which will be forwarded to functor.
    * 
    * an example implemention of functor type is:
    * struct Dumper
    * {
    *     using result_type = void;
    * public:
    *     Dumper(ostream &_os) : os(_os) {}
    * 
    *     template <typename T>
    *     operator (T& insn)
    *     {
    *         insn.dump(os);
    *     }
    * private:
    *     ostream &os;
    * };
    */
    template<typename T, typename ... TArgs >
    typename T::result_type Call(T functor, TArgs&& ... args)
    {
        switch (Type)
        {
        case InsnType::DMACopy:
            return functor(this->DMACopy, std::forward<TArgs>(args)...);
            break;
        
        case InsnType::BufferLS:
            return functor(this->BufferLS, std::forward<TArgs>(args)...);
            break;

        case InsnType::VctrUnary:
            return functor(this->VctrUnary, std::forward<TArgs>(args)...);
            break;

        case InsnType::Li:
            return functor(this->Li, std::forward<TArgs>(args)...);

        default:
            LOG(ERROR) << "undispatched call to NNPUInsn, type code = " << static_cast<int>(Type);
            return typename T::result_type();
            break;
        }
    }

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

struct InsnDumper
{
    using result_type = void;
public:
    template<typename T>
    void operator()(const T& value, std::ostream& os)
    {
        value.Dump(os);
    }
};

}  // namespace nnpu

#endif