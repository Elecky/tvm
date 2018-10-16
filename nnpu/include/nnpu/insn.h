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
    VctrUnary, DMACopy, BufferLS, Li, Stall, Gemm, VctrBinary, VctrDotProd, VctrReduce, VctrImm,
    MatBinary, MatImm, MatReduceRow, MatReduceCol, MatVctr
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

struct StallInsn
{
public:
    StallInsn() = default;

    void Dump(std::ostream& os) const;
};

struct GemmInsn
{
public:
    GemmInsn() = default;

    GemmInsn(uint32_t _nRowOut, uint32_t _factor, uint32_t _nColOut, 
             uint32_t _outAddrReg, uint32_t _in1AddrReg, uint32_t _in2AddrReg, ModeCode _mode)
            : NRowOut(_nRowOut), Factor(_factor), NColOut(_nColOut),
              OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
              Mode(_mode)
    {}

    // the following 3 field is not Imm nor register operand, 
    // they are part of the instruction encoding, since only limited kinds of Gemm is supported,
    // it's possible to use a few bits to encode the gemm operand shape in real use, 
    // here to ease the coding of simulator, we use 3 uint32 instead.
    uint32_t NRowOut;
    uint32_t Factor;
    uint32_t NColOut;

    uint32_t OutAddrReg;
    uint32_t In1AddrReg;
    uint32_t In2AddrReg;

    ModeCode Mode;

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;
};

enum class VctrBinaryOp { Add, Sub, Div, Mul, GTM /* greater than merge */ };
const char* ToString(VctrBinaryOp value);

struct VctrBinaryInsn
{
public:
    VctrBinaryInsn() = default;

    VctrBinaryInsn(VctrBinaryOp _op, uint32_t _outAddrReg, uint32_t _in1AddrReg, 
        uint32_t _in2AddrReg, uint32_t _size, ModeCode _mode) :
        Op(_op), OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
        Size(_size), Mode(_mode)
    {}

    VctrBinaryOp Op;

    uint32_t OutAddrReg;
    uint32_t In1AddrReg;
    uint32_t In2AddrReg;

    // this is not a register nor an immediate, its part of insn encoding.
    uint32_t Size;
    ModeCode Mode;

    void Dump(std::ostream& os) const;
};

enum class VctrImmOp { Add, Sub, Div, Mul,GTM/* greater than merge */,RSub };
const char* ToString(VctrImmOp value);

struct VctrImmInsn
{
public:
    VctrImmInsn() = default;

    VctrImmInsn(VctrImmOp _op, uint32_t _outAddrReg, uint32_t _inAddrReg, 
        double _Imm, uint32_t _size, ModeCode _mode) :
        Op(_op), OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Imm(_Imm),
        Size(_size), Mode(_mode)
    {}

    VctrImmOp Op;

    uint32_t OutAddrReg;
    uint32_t InAddrReg;
    double Imm;

    uint32_t Size;
    ModeCode Mode;

    void Dump(std::ostream& os) const;
};

enum class MatImmOp { Add,Mul,RSub/* greater than merge */ };
const char* ToString(MatImmOp value);
struct MatImmInsn
{
public:
    MatImmInsn() = default;

    MatImmInsn(MatImmOp _op, uint32_t _outAddrReg, uint32_t _inAddrReg, 
        double _Imm,uint32_t _nRow,uint32_t _nCol, ModeCode _mode) :
        Op(_op), OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Imm(_Imm),
        nRow(_nRow), nCol(_nCol), Mode(_mode)
    {}

    MatImmOp Op;

    uint32_t OutAddrReg;
    uint32_t InAddrReg;

    double Imm;
    uint32_t nRow;
    uint32_t nCol;
    ModeCode Mode;

    void Dump(std::ostream& os) const;
};

struct VctrDotProdInsn
{
public:
    VctrDotProdInsn() = default;

    VctrDotProdInsn(uint32_t _outAddrReg, uint32_t _in1AddrReg, 
        uint32_t _in2AddrReg, uint32_t _size, ModeCode _mode) :
        OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
        Size(_size), Mode(_mode)
    {}

    uint32_t OutAddrReg;
    uint32_t In1AddrReg;
    uint32_t In2AddrReg;

    // this is not a register nor an immediate, its part of insn encoding.
    uint32_t Size;
    ModeCode Mode;

    void Dump(std::ostream& os) const;
};

enum class ReduceOp { Sum, Max, Min };
const char* ToString(ReduceOp value);

struct VctrReduceInsn
{
public:
    VctrReduceInsn() = default;

    VctrReduceInsn(uint32_t _outAddrReg, uint32_t _inAddrReg, 
                   ReduceOp _op, uint32_t _size, ModeCode _mode) :
        OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Op(_op), Size(_size), Mode(_mode)
    {}

    uint32_t OutAddrReg;
    uint32_t InAddrReg;

    // this is not a register nor an immediate, its part of insn encoding.
    ReduceOp Op;
    uint32_t Size;
    ModeCode Mode;

    void Dump(std::ostream& os) const;
};

enum class MatBinaryOp { Add, Sub, Mul };
const char* ToString(MatBinaryOp value);

struct MatBinaryInsn
{
public:
    MatBinaryInsn() = default;

    MatBinaryInsn(uint32_t _outAddrReg, uint32_t _in1AddrReg, uint32_t _in2AddrReg,
                  MatBinaryOp _op, uint32_t _size, ModeCode _mode) :
        OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
        Op(_op), Size(_size), Mode(_mode)
    {}

    uint32_t OutAddrReg;
    uint32_t In1AddrReg;
    uint32_t In2AddrReg;

    // the following fileds are part of insn encoding.
    MatBinaryOp Op;
    uint32_t Size;  // the total elements in both matrix.
    ModeCode Mode;

    void Dump(std::ostream& os) const;
};

struct MatReduceRowInsn
{
public:
    MatReduceRowInsn() = default;

    MatReduceRowInsn(uint32_t _outAddrReg, uint32_t _inAddrReg, ReduceOp _op,
        uint32_t _nRow, uint32_t _nCol, ModeCode _mode) :
        OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Op(_op),
        NRow(_nRow), NCol(_nCol), Mode(_mode)
    {}

    uint32_t OutAddrReg;
    uint32_t InAddrReg;

    // the following fileds are part of insn encoding.
    ReduceOp Op;
    uint32_t NRow, NCol;
    ModeCode Mode;

    void Dump(std::ostream &os) const;
};

struct MatReduceColInsn
{
public:
    MatReduceColInsn() = default;

    MatReduceColInsn(uint32_t _outAddrReg, uint32_t _inAddrReg, ReduceOp _op,
        uint32_t _nRow, uint32_t _nCol, ModeCode _mode) :
        OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Op(_op),
        NRow(_nRow), NCol(_nCol), Mode(_mode)
    {}

    uint32_t OutAddrReg;
    uint32_t InAddrReg;

    // the following fileds are part of insn encoding.
    ReduceOp Op;
    uint32_t NRow, NCol;
    ModeCode Mode;

    void Dump(std::ostream &os) const;
};

enum class MatVctrOp { Add, Sub, Mul };
const char* ToString(MatVctrOp value);

struct MatVctrInsn
{
public:
    MatVctrInsn() = default;

    MatVctrInsn(uint32_t _outAddrReg, uint32_t _matAddrReg, uint32_t _vctrAddrReg,
                MatVctrOp _op, uint32_t _nRow, uint32_t _nCol, ModeCode _mode) :
        OutAddrReg(_outAddrReg), MatAddrReg(_matAddrReg), VctrAddrReg(_vctrAddrReg),
        Op(_op), NRow(_nRow), NCol(_nCol), Mode(_mode)
    {}

    uint32_t OutAddrReg;
    uint32_t MatAddrReg;
    uint32_t VctrAddrReg;

    MatVctrOp Op;
    uint32_t NRow, NCol;
    ModeCode Mode;

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

        VctrBinaryInsn VctrBinary;

        VctrImmInsn VctrImm;

        MatImmInsn MatImm;

        LiInsn Li;

        StallInsn stall;

        GemmInsn Gemm;

        VctrDotProdInsn VctrDotProd;

        VctrReduceInsn VctrReduce;

        MatBinaryInsn MatBinary;

        MatReduceRowInsn MatReduceRow;

        MatReduceColInsn MatReduceCol;

        MatVctrInsn MatVctr;
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

        case InsnType::Stall:
            return functor(this->stall, std::forward<TArgs>(args)...);

        case InsnType::Gemm:
            return functor(this->Gemm, std::forward<TArgs>(args)...);

        case InsnType::VctrBinary:
            return functor(this->VctrBinary, std::forward<TArgs>(args)...);
        
        case InsnType::VctrImm:
            return functor(this->VctrImm, std::forward<TArgs>(args)...);

        case InsnType::MatImm:
            return functor(this->MatImm, std::forward<TArgs>(args)...);

        case InsnType::VctrDotProd:
            return functor(this->VctrDotProd, std::forward<TArgs>(args)...);

        case InsnType::VctrReduce:
            return functor(this->VctrReduce, std::forward<TArgs>(args)...);
        
        case InsnType::MatBinary:
            return functor(this->MatBinary, std::forward<TArgs>(args)...);

        case InsnType::MatReduceRow:
            return functor(this->MatReduceRow, std::forward<TArgs>(args)...);

        case InsnType::MatReduceCol:
            return functor(this->MatReduceCol, std::forward<TArgs>(args)...);

        case InsnType::MatVctr:
            return functor(this->MatVctr, std::forward<TArgs>(args)...);

        default:
            LOG(ERROR) << "undispatched call to NNPUInsn, type code = " << static_cast<int>(Type) 
                       << ". please modify NNPUInsn::Call to implement missing dispatch";
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

    NNPUInsn(const StallInsn &_insn) : Type(InsnType::Stall), stall(_insn)
    {}

    NNPUInsn(const GemmInsn &_insn) : Type(InsnType::Gemm), Gemm(_insn)
    {}

    NNPUInsn(const VctrBinaryInsn &_insn) : Type(InsnType::VctrBinary), VctrBinary(_insn)
    {}

    NNPUInsn(const VctrImmInsn &_insn) : Type(InsnType::VctrImm), VctrImm(_insn)
    {}

    NNPUInsn(const MatImmInsn &_insn) : Type(InsnType::MatImm), MatImm(_insn)
    {}
    
    NNPUInsn(const VctrDotProdInsn &_insn) : Type(InsnType::VctrDotProd), VctrDotProd(_insn)
    {}

    NNPUInsn(const VctrReduceInsn &_insn) : Type(InsnType::VctrReduce), VctrReduce(_insn)
    {}

    NNPUInsn(const MatBinaryInsn &_insn) : Type(InsnType::MatBinary), MatBinary(_insn)
    {}

    NNPUInsn(const MatReduceRowInsn &_insn) : Type(InsnType::MatReduceRow), MatReduceRow(_insn)
    {}

    NNPUInsn(const MatReduceColInsn &_insn) : Type(InsnType::MatReduceCol), MatReduceCol(_insn)
    {}

    NNPUInsn(const MatVctrInsn &_insn) : Type(InsnType::MatVctr), MatVctr(_insn)
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