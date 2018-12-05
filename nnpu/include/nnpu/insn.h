/*
the instruction set definitions of nnpu simulator
*/

#ifndef NNPU_INSN_H
#define NNPU_INSN_H

#include <cstdint>
#include <iostream>
#include <dmlc/logging.h>
#include <nnpusim/typedef.h>
#include <nnpusim/common/data_types.h>
#include <unordered_map>

namespace nnpu
{

// the K-V list type to store register values.
using KVList_t = std::unordered_map<regNo_t, reg_t>;
using dst_pair_t = std::pair<bool, regNo_t>;
using branch_off_t = std::pair<bool, reg_t>;

/*
* instruction type, similar to opcode in MIPS instruction
*/
enum class InsnType
{
    VctrUnary, DMACopy, BufferLS, Li, Stall, Gemm, VctrBinary, VctrDotProd, VctrReduce, VctrImm,
    MatBinary, MatImm, MatReduceRow, MatReduceCol, MatVctr, MatRowDot, VctrSclr, BufferCopy,
    Memset, AccMemset, CopyAcc2Buf, NOP, Jump, BEZ, ALUBinary, SclrLoad, SclrStore, BNEZ,
    ALURegImm
};

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
    regNo_t VctrOutAddrReg;
    regNo_t VctrInAddrReg;
    regNo_t ElemCountReg;

    ModeCode Mode;

    /* default constructor */
    VctrUnaryInsn() = default;

    /* constructor */
    VctrUnaryInsn(VctrUnaryOp _op, regNo_t _vctrOutAddrReg, regNo_t _vctrInAddrReg, 
        regNo_t _elemCountReg, ModeCode _mode) :
        Op(_op), VctrOutAddrReg(_vctrOutAddrReg), VctrInAddrReg(_vctrInAddrReg), 
        ElemCountReg(_elemCountReg), Mode(_mode)
    {}

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;

    /*!
     * \brief get the register value map.
     * \return a KVList_t contains all regiseters operands in key.
    */
    KVList_t GetRegMap() const;

    /*!
     * \brief get the destination register of this insn.
     * \ return a pair<bool, regNo_t>, first value is has destination register or not,
     *                                 second value is the destination register No.
    */
    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    /*!
     * \brief check whether this insn is a branch insn,
     *        and get the branch offset of this insn if so.
     * \return a pair<bool, int32_t>, first value indicates whether this insn is a branch,
     *                                second value is the branch offset.
    */
    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
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
    regNo_t HostPhyAddrReg;
    regNo_t HostOffsetReg;
    regNo_t DramAddrReg;
    /* copy size in byte
    */
    regNo_t SizeReg;

    /* default constructor */
    DMACopyInsn() = default;

    /* constructor */
    DMACopyInsn(DMADIR _dir, regNo_t _hostPhyAddrReg, regNo_t _hostOffsetReg, 
        regNo_t _dramAddrReg, regNo_t _sizeReg) :
        Dir(_dir), HostPhyAddrReg(_hostPhyAddrReg), HostOffsetReg(_hostOffsetReg), 
        DramAddrReg(_dramAddrReg), SizeReg(_sizeReg)
    {}

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
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

    regNo_t DramAddrReg;
    regNo_t BufAddrReg;
    /* copy size in byte */
    regNo_t SizeReg;

    /* default constructor */
    BufferLSInsn() = default;

    /* constructor */
    BufferLSInsn(LSDIR _dir, regNo_t _dramAddrReg, regNo_t _bufAddrReg, regNo_t _sizeReg) :
        Dir(_dir), DramAddrReg(_dramAddrReg), BufAddrReg(_bufAddrReg), SizeReg(_sizeReg)
    {}

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct BufferCopyInsn
{
public:
    BufferCopyInsn() = default;

    BufferCopyInsn(regNo_t _dstAddrReg, regNo_t _dstStrideReg, 
                   regNo_t _srcAddrReg, regNo_t _srcStrideReg,
                   regNo_t _nUnitReg, uint32_t _unitBytes) :
        DstAddrReg(_dstAddrReg), DstStrideReg(_dstStrideReg),
        SrcAddrReg(_srcAddrReg), SrcStrideReg(_srcStrideReg),
        NUnitReg(_nUnitReg), UnitBytes(_unitBytes)
    {}

    regNo_t DstAddrReg;
    regNo_t DstStrideReg;
    regNo_t SrcAddrReg;
    regNo_t SrcStrideReg;
    regNo_t NUnitReg;

    uint32_t UnitBytes;

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct MemsetInsn
{
public:
    MemsetInsn() = default;

    MemsetInsn(regNo_t _addrReg, regNo_t _nUnitReg, regNo_t _strideReg,
               ModeCode _mode, double _imm) :
        AddrReg(_addrReg), NUnitReg(_nUnitReg), StrideReg(_strideReg),
        Mode(_mode), Imm(_imm)
    {}

    regNo_t AddrReg;
    regNo_t NUnitReg;
    regNo_t StrideReg;

    ModeCode Mode;  // can only be n or w.
    double Imm;
    
    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

/*!
 * \brief used to init accumulation 
*/
struct AccMemsetInsn
{
public:
    AccMemsetInsn() = default;

    AccMemsetInsn(uint32_t _nRow, uint32_t _nCol, 
                  regNo_t _addrReg, regNo_t _rowStrideReg,
                  ModeCode _mode, double _imm) :
        AddrReg(_addrReg), RowStrideReg(_rowStrideReg),
        NRow(_nRow), NCol(_nCol), Mode(_mode), Imm(_imm)
    {}

    regNo_t AddrReg;
    regNo_t RowStrideReg;

    uint32_t NRow, NCol;
    ModeCode Mode;  // can only be n or w.
    double Imm;  // currently, accumulation memory is only inited by zero.

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

/*!
* \brief Load Immediate Insn,
*        assign an immediate value to a register
*/
struct LiInsn
{
public:
    regNo_t ResReg;
    reg_t Imm;

    /* default constructor */
    LiInsn() = default;

    LiInsn(regNo_t _resReg, reg_t _imm) : ResReg(_resReg), Imm(_imm)
    {}

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {true, ResReg};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct StallInsn
{
public:
    StallInsn() = default;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct GemmInsn
{
public:
    GemmInsn() = default;

    GemmInsn(uint32_t _nRowOut, uint32_t _factor, uint32_t _nColOut, 
             regNo_t _outAddrReg, regNo_t _outRowStrideReg,
             regNo_t _in1AddrReg, regNo_t _in1RowStrideReg,
             regNo_t _in2AddrReg, regNo_t _in2RowStrideReg, ModeCode _mode,
             bool _toAccBuf, bool _doAcc)
            : NRowOut(_nRowOut), Factor(_factor), NColOut(_nColOut),
              OutAddrReg(_outAddrReg), OutRowStrideReg(_outRowStrideReg),
              In1AddrReg(_in1AddrReg), In1RowStrideReg(_in1RowStrideReg),
              In2AddrReg(_in2AddrReg), In2RowStrideReg(_in2RowStrideReg),
              ToAccBuf(_toAccBuf), DoAcc(_doAcc), Mode(_mode)
    {}

    // the following 3 field is not Imm nor register operand, 
    // they are part of the instruction encoding, since only limited kinds of Gemm is supported,
    // it's possible to use a few bits to encode the gemm operand shape in real use, 
    // here to ease the coding of simulator, we use 3 uint32 instead.
    uint32_t NRowOut;
    uint32_t Factor;
    uint32_t NColOut;

    regNo_t OutAddrReg;
    regNo_t OutRowStrideReg;
    regNo_t In1AddrReg;
    regNo_t In1RowStrideReg;
    regNo_t In2AddrReg;
    regNo_t In2RowStrideReg;

    bool ToAccBuf;  // write to accumulation buffer or not.
    bool DoAcc;  // do accumulation or not.
    ModeCode Mode;

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

enum class VctrBinaryOp { Add, Sub, Div, Mul, GTM /* greater than merge */ };
const char* ToString(VctrBinaryOp value);

struct VctrBinaryInsn
{
public:
    VctrBinaryInsn() = default;

    VctrBinaryInsn(VctrBinaryOp _op, regNo_t _outAddrReg, regNo_t _in1AddrReg, 
        regNo_t _in2AddrReg, uint32_t _size, ModeCode _mode) :
        Op(_op), OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
        Size(_size), Mode(_mode)
    {}

    VctrBinaryOp Op;

    regNo_t OutAddrReg;
    regNo_t In1AddrReg;
    regNo_t In2AddrReg;

    // this is not a register nor an immediate, its part of insn encoding.
    uint32_t Size;
    ModeCode Mode;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

enum class VctrImmOp { Add, Sub, Div, Mul,GTM/* greater than merge */, RSub , RDiv };
const char* ToString(VctrImmOp value);

struct VctrImmInsn
{
public:
    VctrImmInsn() = default;

    VctrImmInsn(VctrImmOp _op, regNo_t _outAddrReg, regNo_t _inAddrReg, 
        double _Imm, uint32_t _size, ModeCode _mode) :
        Op(_op), OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Imm(_Imm),
        Size(_size), Mode(_mode)
    {}

    VctrImmOp Op;

    regNo_t OutAddrReg;
    regNo_t InAddrReg;
    double Imm;

    uint32_t Size;
    ModeCode Mode;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

enum class MatImmOp { Add,Mul,RSub/* greater than merge */ };
const char* ToString(MatImmOp value);
struct MatImmInsn
{
public:
    MatImmInsn() = default;

    MatImmInsn(MatImmOp _op, regNo_t _outAddrReg, regNo_t _inAddrReg, 
        double _Imm,uint32_t _nRow, uint32_t _nCol, ModeCode _mode) :
        Op(_op), OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Imm(_Imm),
        nRow(_nRow), nCol(_nCol), Mode(_mode)
    {}

    MatImmOp Op;

    regNo_t OutAddrReg;
    regNo_t InAddrReg;

    double Imm;
    uint32_t nRow;
    uint32_t nCol;
    ModeCode Mode;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct VctrDotProdInsn
{
public:
    VctrDotProdInsn() = default;

    VctrDotProdInsn(regNo_t _outAddrReg, regNo_t _in1AddrReg, 
        regNo_t _in2AddrReg, uint32_t _size, ModeCode _mode) :
        OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
        Size(_size), Mode(_mode)
    {}

    regNo_t OutAddrReg;
    regNo_t In1AddrReg;
    regNo_t In2AddrReg;

    // this is not a register nor an immediate, its part of insn encoding.
    uint32_t Size;
    ModeCode Mode;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

enum class ReduceOp { Sum, Max, Min };
const char* ToString(ReduceOp value);

struct VctrReduceInsn
{
public:
    VctrReduceInsn() = default;

    VctrReduceInsn(regNo_t _outAddrReg, regNo_t _inAddrReg, 
                   ReduceOp _op, uint32_t _size, ModeCode _mode) :
        OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Op(_op), Size(_size), Mode(_mode)
    {}

    regNo_t OutAddrReg;
    regNo_t InAddrReg;

    // this is not a register nor an immediate, its part of insn encoding.
    ReduceOp Op;
    uint32_t Size;
    ModeCode Mode;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

enum class MatBinaryOp { Add, Sub, Mul };
const char* ToString(MatBinaryOp value);

struct MatBinaryInsn
{
public:
    MatBinaryInsn() = default;

    MatBinaryInsn(regNo_t _outAddrReg, regNo_t _in1AddrReg, regNo_t _in2AddrReg,
                  regNo_t _outRowStrideReg, regNo_t _in1RowStrideReg, regNo_t _in2RowStrideReg,
                  MatBinaryOp _op, uint32_t _nRow, uint32_t _nCol, ModeCode _mode) :
        OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
        OutRowStrideReg(_outRowStrideReg), In1RowStrideReg(_in1RowStrideReg), 
        In2RowStrideReg(_in2RowStrideReg), Op(_op), NRow(_nRow), NCol(_nCol), Mode(_mode)
    {}

    regNo_t OutAddrReg;
    regNo_t In1AddrReg;
    regNo_t In2AddrReg;

    regNo_t OutRowStrideReg;
    regNo_t In1RowStrideReg;
    regNo_t In2RowStrideReg;

    // the following fileds are part of insn encoding.
    MatBinaryOp Op;
    uint32_t NRow, NCol;  // the elements in all 3 matrix.
    ModeCode Mode;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct MatReduceRowInsn
{
public:
    MatReduceRowInsn() = default;

    MatReduceRowInsn(regNo_t _outAddrReg, regNo_t _inAddrReg, regNo_t _inRowStrideReg,
                     ReduceOp _op, uint32_t _nRow, uint32_t _nCol, 
                     bool _toAccBuf, bool _doAcc, ModeCode _mode) :
        OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), InRowStrideReg(_inRowStrideReg),
        Op(_op), NRow(_nRow), NCol(_nCol), ToAccBuf(_toAccBuf), DoAcc(_doAcc), Mode(_mode)
    {}

    regNo_t OutAddrReg;
    regNo_t InAddrReg;
    regNo_t InRowStrideReg;

    // the following fileds are part of insn encoding.
    ReduceOp Op;
    uint32_t NRow, NCol;
    bool ToAccBuf;   // write to accumulation buffer or not, if not write to scratchpad.
    bool DoAcc;  // do accumulation or not, only valid when ToAccBuf is true.
    ModeCode Mode;

    void Dump(std::ostream &os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct MatReduceColInsn
{
public:
    MatReduceColInsn() = default;

    MatReduceColInsn(regNo_t _outAddrReg, regNo_t _inAddrReg, ReduceOp _op,
        uint32_t _nRow, uint32_t _nCol, ModeCode _mode) :
        OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Op(_op),
        NRow(_nRow), NCol(_nCol), Mode(_mode)
    {}

    regNo_t OutAddrReg;
    regNo_t InAddrReg;

    // the following fileds are part of insn encoding.
    ReduceOp Op;
    uint32_t NRow, NCol;
    ModeCode Mode;

    void Dump(std::ostream &os) const;
    
    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

enum class MatVctrOp { Add, Sub, Mul };
const char* ToString(MatVctrOp value);

struct MatVctrInsn
{
public:
    MatVctrInsn() = default;

    MatVctrInsn(regNo_t _outAddrReg, regNo_t _outRowStrideReg, 
                regNo_t _matAddrReg, regNo_t _matRowStrideReg,
                regNo_t _vctrAddrReg, MatVctrOp _op, uint32_t _nRow, uint32_t _nCol, 
                ModeCode _mode) :
        OutAddrReg(_outAddrReg), OutRowStrideReg(_outRowStrideReg), 
        MatAddrReg(_matAddrReg), MatRowStrideReg(_matRowStrideReg),
        VctrAddrReg(_vctrAddrReg), Op(_op), NRow(_nRow), NCol(_nCol), Mode(_mode)
    {}

    regNo_t OutAddrReg;
    regNo_t OutRowStrideReg;
    regNo_t MatAddrReg;
    regNo_t MatRowStrideReg;
    regNo_t VctrAddrReg;

    MatVctrOp Op;
    uint32_t NRow, NCol;
    ModeCode Mode;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

/*!
 * \brief calc the dot product of two input matrix on every row.
*/
struct MatRowDotInsn
{
public:
    MatRowDotInsn() = default;

    MatRowDotInsn(regNo_t _outAddrReg, regNo_t _in1AddrReg, regNo_t _in1RowStrideReg,
                  regNo_t _in2AddrReg, regNo_t _in2RowStrideReg,
                  uint32_t _nRow, uint32_t _nCol, 
                  bool _toAccBuf, bool _doAcc, ModeCode _mode) :
        OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In1RowStrideReg(_in1RowStrideReg),
        In2AddrReg(_in2AddrReg), In2RowStrideReg(_in2RowStrideReg),
        NRow(_nRow), NCol(_nCol), toAccBuf(_toAccBuf), doAcc(_doAcc), Mode(_mode)
    {}

    regNo_t OutAddrReg;
    regNo_t In1AddrReg;
    regNo_t In1RowStrideReg;
    regNo_t In2AddrReg;
    regNo_t In2RowStrideReg;

    // the following fileds are part of insn encoding.
    uint32_t NRow, NCol;  // the total elements in both matrix.
    bool toAccBuf;  // write to accumulation buffer or scratchpad?
    bool doAcc;  // do an accumulation?
    ModeCode Mode;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

enum class VctrSclrOp { Add, Sub, Div, Mul,GTM/* greater than merge */, RSub , RDiv };
const char* ToString(VctrSclrOp value);

struct VctrSclrInsn
{
public:
    VctrSclrInsn() = default;

    VctrSclrInsn(regNo_t _outAddrReg, regNo_t _vctrAddrReg, regNo_t _sclrAddrReg,
                 uint32_t _size, VctrSclrOp _op, ModeCode _mode) :
        OutAddrReg(_outAddrReg), VctrAddrReg(_vctrAddrReg), SclrAddrReg(_sclrAddrReg),
        Op(_op), Size(_size), Mode(_mode)
    {}

    regNo_t OutAddrReg;
    regNo_t VctrAddrReg;
    regNo_t SclrAddrReg;

    // the following fileds are part of insn encoding.
    VctrSclrOp Op;
    uint32_t Size;  // the total elements in both matrix.
    ModeCode Mode;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct CopyAcc2BufInsn
{
public:
    CopyAcc2BufInsn() = default;

    CopyAcc2BufInsn(regNo_t _dstAddrReg, regNo_t _dstStrideReg, 
                   regNo_t _srcAddrReg, regNo_t _srcStrideReg,
                   regNo_t _nUnitReg, uint32_t _unitBytes) :
        DstAddrReg(_dstAddrReg), DstStrideReg(_dstStrideReg),
        SrcAddrReg(_srcAddrReg), SrcStrideReg(_srcStrideReg),
        NUnitReg(_nUnitReg), UnitBytes(_unitBytes)
    {}

    regNo_t DstAddrReg;
    regNo_t DstStrideReg;
    regNo_t SrcAddrReg;
    regNo_t SrcStrideReg;
    regNo_t NUnitReg;

    uint32_t UnitBytes;

    /*!
    * \brief dump the string representation of this instruction into ostream
    * \param os: the stream to which to dump.
    */
    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct NOPInsn
{
    NOPInsn() = default;

    void Dump(std::ostream& os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct JumpInsn
{
    JumpInsn() = default;

    JumpInsn(reg_t _offset):
        Offset(_offset)
    {}

    reg_t Offset;

    void Dump(std::ostream &os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        // Jump is not a branch. and Jump is expilictly handled in code.
        return {false, 0};
    }
};

struct BEZInsn
{
    BEZInsn() = default;

    BEZInsn(reg_t _offset, regNo_t _condReg) :
        Offset(_offset), CondReg(_condReg)
    {}

    reg_t Offset;
    regNo_t CondReg;

    void Dump(std::ostream &os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {true, Offset};
    }
};

/*!
 * \brief branch not equal to Zero insn.
*/
struct BNEZInsn
{
    BNEZInsn() = default;

    BNEZInsn(reg_t _offset, regNo_t _condReg) :
        Offset(_offset), CondReg(_condReg)
    {}

    reg_t Offset;
    regNo_t CondReg;

    void Dump(std::ostream &os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {true, Offset};
    }
};

enum class ALUBinaryOp { Add, Sub, Mul, DivU, ModU, SLTU /*set less than unsigned*/, 
                         SEQ, XOR, And, Or };
const char* ToString(ALUBinaryOp op);

struct ALUBinaryInsn
{
    ALUBinaryInsn() = default;

    ALUBinaryInsn(regNo_t _rdReg, regNo_t _rsReg, regNo_t _rtReg, ALUBinaryOp _op);

    regNo_t RdReg;
    regNo_t RsReg, RtReg;

    ALUBinaryOp Op;

    void Dump(std::ostream &os) const;

    /*!
     * \brief get the register value map.
     * \return a KVList_t contains all regiseters operands in key.
    */
    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {true, RdReg};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

enum class ALURegImmOp { AddIU, ISubU /*Imm subtract Register*/, MulIU, DivIU, 
                         ModIU, SLTIU, SEQI, XORI, AndI, OrI };
const char * ToString(ALURegImmOp op);

struct ALURegImmInsn
{
    ALURegImmInsn() = default;

    ALURegImmInsn(regNo_t _rdReg, regNo_t _rsReg, reg_t _imm, ALURegImmOp _op);

    regNo_t RdReg;
    regNo_t RsReg;

    reg_t Imm;
    ALURegImmOp Op;

    void Dump(std::ostream &os) const;

    /*!
     * \brief get the register value map.
     * \return a KVList_t contains all regiseters operands in key.
    */
    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {true, RdReg};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct SclrLoadInsn
{
    SclrLoadInsn() = default;

    SclrLoadInsn(regNo_t _rdReg, regNo_t _addrReg, reg_t _offset);

    regNo_t RdReg;
    regNo_t AddrReg;

    reg_t Offset;

    void Dump(std::ostream &os) const;

    /*!
     * \brief get the register value map.
     * \return a KVList_t contains all regiseters operands in key.
    */
    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {true, RdReg};
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

struct SclrStoreInsn
{
    SclrStoreInsn() = default;

    SclrStoreInsn(regNo_t _rsReg, regNo_t _addrReg, reg_t _offset);

    regNo_t RsReg;
    regNo_t AddrReg;

    reg_t Offset;

    void Dump(std::ostream &os) const;

    KVList_t GetRegMap() const;

    inline dst_pair_t GetDstReg() const
    {
        return {false, 0};  // store insn has no register destination.
    }

    inline branch_off_t GetBranchOffset() const
    {
        return {false, 0};
    }
};

/*
* \brief nnpu instruction struct, contains a union of actual instructions, 
*        and a InsnType field.
*/
struct NNPUInsn {
public:
    NNPUInsn(InsnType type) : Type(type) {}

    InsnType Type;

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

        MatRowDotInsn MatRowDot;

        VctrSclrInsn VctrSclr;

        BufferCopyInsn BufferCopy;

        MemsetInsn Memset;

        CopyAcc2BufInsn CopyAcc2Buf;

        AccMemsetInsn AccMemset;

        NOPInsn NOP;

        JumpInsn Jump;

        BEZInsn BEZ;

        BNEZInsn BNEZ;

        ALUBinaryInsn ALUBinary;

        SclrLoadInsn SclrLoad;

        SclrStoreInsn SclrStore;

        ALURegImmInsn ALURegImm;
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
    typename T::result_type Call(T functor, TArgs&& ... args) const
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

        case InsnType::MatRowDot:
            return functor(this->MatRowDot, std::forward<TArgs>(args)...);

        case InsnType::VctrSclr:
            return functor(this->VctrSclr, std::forward<TArgs>(args)...);

        case InsnType::BufferCopy:
            return functor(this->BufferCopy, std::forward<TArgs>(args)...);

        case InsnType::Memset:
            return functor(this->Memset, std::forward<TArgs>(args)...);

        case InsnType::CopyAcc2Buf:
            return functor(this->CopyAcc2Buf, std::forward<TArgs>(args)...);

        case InsnType::AccMemset:
            return functor(this->AccMemset, std::forward<TArgs>(args)...);

        case InsnType::NOP:
            return functor(this->NOP, std::forward<TArgs>(args)...);
        
        case InsnType::Jump:
            return functor(this->Jump, std::forward<TArgs>(args)...);

        case InsnType::BEZ:
            return functor(this->BEZ, std::forward<TArgs>(args)...);

        case InsnType::ALUBinary:
            return functor(this->ALUBinary, std::forward<TArgs>(args)...);

        case InsnType::SclrLoad:
            return functor(this->SclrLoad, std::forward<TArgs>(args)...);

        case InsnType::SclrStore:
            return functor(this->SclrStore, std::forward<TArgs>(args)...);

        case InsnType::BNEZ:
            return functor(this->BNEZ, std::forward<TArgs>(args)...);

        case InsnType::ALURegImm:
            return functor(this->ALURegImm, std::forward<TArgs>(args)...);

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

    NNPUInsn(const MatRowDotInsn &_insn) : Type(InsnType::MatRowDot), MatRowDot(_insn)
    {}

    NNPUInsn(const VctrSclrInsn &_insn) : Type(InsnType::VctrSclr), VctrSclr(_insn)
    {}

    NNPUInsn(const BufferCopyInsn &_insn) : Type(InsnType::BufferCopy), BufferCopy(_insn)
    {}

    NNPUInsn(const MemsetInsn &_insn) : Type(InsnType::Memset), Memset(_insn)
    {}

    NNPUInsn(const CopyAcc2BufInsn &_insn) : Type(InsnType::CopyAcc2Buf), CopyAcc2Buf(_insn)
    {}

    NNPUInsn(const AccMemsetInsn &_insn) : Type(InsnType::AccMemset), AccMemset(_insn)
    {}

    NNPUInsn(const NOPInsn &_insn) : Type(InsnType::NOP)
    {}

    NNPUInsn(const JumpInsn &_insn) : Type(InsnType::Jump), Jump(_insn)
    {}

    NNPUInsn(const BEZInsn &_insn) : Type(InsnType::BEZ), BEZ(_insn)
    {}

    NNPUInsn(const ALUBinaryInsn &_insn) : Type(InsnType::ALUBinary), ALUBinary(_insn)
    {}

    NNPUInsn(const SclrLoadInsn &_insn) : Type(InsnType::SclrLoad), SclrLoad(_insn)
    {}

    NNPUInsn(const SclrStoreInsn &_insn) : Type(InsnType::SclrStore), SclrStore(_insn)
    {}

    NNPUInsn(const BNEZInsn &_insn) : Type(InsnType::BNEZ), BNEZ(_insn)
    {}

    NNPUInsn(const ALURegImmInsn &_insn) : Type(InsnType::ALURegImm), ALURegImm(_insn)
    {}
};

struct InsnDumper
{
    using result_type = void;
public:
    template<typename T>
    inline void operator()(const T& value, std::ostream& os)
    {
        value.Dump(os);
    }
};

}  // namespace nnpu

#endif