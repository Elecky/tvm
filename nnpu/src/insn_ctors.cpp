#include <nnpu/insn.h>
#include <string>
#include <stdexcept>

using std::ostream;
using std::string;

namespace nnpu
{

ALUBinaryInsn::ALUBinaryInsn(regNo_t _rdReg, regNo_t _rsReg, regNo_t _rtReg, ALUBinaryOp _op) :
    RdReg(_rdReg), RsReg(_rsReg), RtReg(_rtReg), Op(_op)
{}

SclrLoadInsn::SclrLoadInsn(regNo_t _rdReg, regNo_t _addrReg, reg_t _offset) :
    RdReg(_rdReg), AddrReg(_addrReg), Offset(_offset)
{}

SclrStoreInsn::SclrStoreInsn(regNo_t _rsReg, regNo_t _addrReg, reg_t _offset) :
   RsReg(_rsReg),  AddrReg(_addrReg), Offset(_offset)
{}

ALURegImmInsn::ALURegImmInsn(regNo_t _rdReg, regNo_t _rsReg, reg_t _imm, ALURegImmOp _op) :
    RdReg(_rdReg), RsReg(_rsReg), Imm(_imm), Op(_op)
{}

VctrUnaryInsn::VctrUnaryInsn(
        VctrUnaryOp _op, regNo_t _vctrOutAddrReg, 
        regNo_t _vctrInAddrReg, 
        uint32_t _size, ModeCode _mode) :
    Op(_op), VctrOutAddrReg(_vctrOutAddrReg), 
    VctrInAddrReg(_vctrInAddrReg), 
    Size(_size), Mode(_mode)
{}

DMACopyInsn::DMACopyInsn(
        DMADIR _dir, regNo_t _hostPhyAddrReg, regNo_t _hostOffsetReg, 
        regNo_t _dramAddrReg, regNo_t _sizeReg) :
    Dir(_dir), HostPhyAddrReg(_hostPhyAddrReg), HostOffsetReg(_hostOffsetReg), 
    DramAddrReg(_dramAddrReg), SizeReg(_sizeReg)
{}

BufferLSInsn::BufferLSInsn(
        LSDIR _dir, regNo_t _dramAddrReg, 
        regNo_t _bufAddrReg, regNo_t _sizeReg) :
    Dir(_dir), DramAddrReg(_dramAddrReg), 
    BufAddrReg(_bufAddrReg), SizeReg(_sizeReg)
{}

BufferCopyInsn::BufferCopyInsn(
        regNo_t _dstAddrReg, regNo_t _dstStrideReg, 
        regNo_t _srcAddrReg, regNo_t _srcStrideReg,
        regNo_t _nUnitReg, uint32_t _unitBytes) :
    DstAddrReg(_dstAddrReg), DstStrideReg(_dstStrideReg),
    SrcAddrReg(_srcAddrReg), SrcStrideReg(_srcStrideReg),
    NUnitReg(_nUnitReg), UnitBytes(_unitBytes)
{}

MemsetInsn::MemsetInsn(
        regNo_t _addrReg, regNo_t _nUnitReg, 
        regNo_t _strideReg,
        ModeCode _mode, double _imm) :
    AddrReg(_addrReg), NUnitReg(_nUnitReg), 
    StrideReg(_strideReg),
    Mode(_mode), Imm(_imm)
{}

AccMemsetInsn::AccMemsetInsn(
        uint32_t _nRow, uint32_t _nCol, 
        regNo_t _addrReg, regNo_t _rowStrideReg,
        ModeCode _mode, double _imm) :
    AddrReg(_addrReg), RowStrideReg(_rowStrideReg),
    NRow(_nRow), NCol(_nCol), 
    Mode(_mode), Imm(_imm)
{}

LiInsn::LiInsn(
        regNo_t _resReg, reg_t _imm) : 
    ResReg(_resReg), Imm(_imm)
{}

GemmInsn::GemmInsn(
        uint32_t _nRowOut, uint32_t _factor, uint32_t _nColOut, 
        regNo_t _outAddrReg, regNo_t _outRowStrideReg,
        regNo_t _in1AddrReg, regNo_t _in1RowStrideReg,
        regNo_t _in2AddrReg, regNo_t _in2RowStrideReg, ModeCode _mode,
        bool _toAccBuf, bool _doAcc) : 
    NRowOut(_nRowOut), Factor(_factor), NColOut(_nColOut),
    OutAddrReg(_outAddrReg), OutRowStrideReg(_outRowStrideReg),
    In1AddrReg(_in1AddrReg), In1RowStrideReg(_in1RowStrideReg),
    In2AddrReg(_in2AddrReg), In2RowStrideReg(_in2RowStrideReg),
    ToAccBuf(_toAccBuf), DoAcc(_doAcc), Mode(_mode)
{}

VctrBinaryInsn::VctrBinaryInsn(
        VctrBinaryOp _op, regNo_t _outAddrReg, 
        regNo_t _in1AddrReg, regNo_t _in2AddrReg, 
        uint32_t _size, ModeCode _mode) :
    Op(_op), OutAddrReg(_outAddrReg), 
    In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
    Size(_size), Mode(_mode)
{}

VctrImmInsn::VctrImmInsn(
        VctrImmOp _op, regNo_t _outAddrReg, 
        regNo_t _inAddrReg, double _Imm, 
        uint32_t _size, ModeCode _mode) :
    Op(_op), OutAddrReg(_outAddrReg), 
    InAddrReg(_inAddrReg), Imm(_Imm),
    Size(_size), Mode(_mode)
{}

MatImmInsn::MatImmInsn(
        MatImmOp _op, regNo_t _outAddrReg, 
        regNo_t _inAddrReg, double _Imm,
        uint32_t _nRow, uint32_t _nCol, ModeCode _mode) :
    Op(_op), OutAddrReg(_outAddrReg), 
    InAddrReg(_inAddrReg), Imm(_Imm),
    nRow(_nRow), nCol(_nCol), Mode(_mode)
{}

VctrDotProdInsn::VctrDotProdInsn(
        regNo_t _outAddrReg, regNo_t _in1AddrReg, 
        regNo_t _in2AddrReg, uint32_t _size, ModeCode _mode) :
    OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
    Size(_size), Mode(_mode)
{}

VctrReduceInsn::VctrReduceInsn(
        regNo_t _outAddrReg, regNo_t _inAddrReg, 
        ReduceOp _op, uint32_t _size, ModeCode _mode) :
    OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), 
    Op(_op), Size(_size), Mode(_mode)
{}

MatBinaryInsn::MatBinaryInsn(
        regNo_t _outAddrReg, regNo_t _in1AddrReg, regNo_t _in2AddrReg,
        regNo_t _outRowStrideReg, regNo_t _in1RowStrideReg, regNo_t _in2RowStrideReg,
        MatBinaryOp _op, uint32_t _nRow, uint32_t _nCol, ModeCode _mode) :
    OutAddrReg(_outAddrReg), In1AddrReg(_in1AddrReg), In2AddrReg(_in2AddrReg),
    OutRowStrideReg(_outRowStrideReg), In1RowStrideReg(_in1RowStrideReg), 
    In2RowStrideReg(_in2RowStrideReg), Op(_op), NRow(_nRow), NCol(_nCol), Mode(_mode)
{}

MatReduceRowInsn::MatReduceRowInsn(
        regNo_t _outAddrReg, regNo_t _inAddrReg, regNo_t _inRowStrideReg,
        ReduceOp _op, uint32_t _nRow, uint32_t _nCol, 
        bool _toAccBuf, bool _doAcc, ModeCode _mode) :
    OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), InRowStrideReg(_inRowStrideReg),
    Op(_op), NRow(_nRow), NCol(_nCol), ToAccBuf(_toAccBuf), DoAcc(_doAcc), Mode(_mode)
{}

MatReduceColInsn::MatReduceColInsn(
        regNo_t _outAddrReg, regNo_t _inAddrReg, ReduceOp _op,
        uint32_t _nRow, uint32_t _nCol, ModeCode _mode) :
    OutAddrReg(_outAddrReg), InAddrReg(_inAddrReg), Op(_op),
    NRow(_nRow), NCol(_nCol), Mode(_mode)
{}

MatVctrInsn::MatVctrInsn(
        regNo_t _outAddrReg, regNo_t _outRowStrideReg, 
        regNo_t _matAddrReg, regNo_t _matRowStrideReg,
        regNo_t _vctrAddrReg, MatVctrOp _op, 
        uint32_t _nRow, uint32_t _nCol, 
        ModeCode _mode) :
    OutAddrReg(_outAddrReg), OutRowStrideReg(_outRowStrideReg), 
    MatAddrReg(_matAddrReg), MatRowStrideReg(_matRowStrideReg),
    VctrAddrReg(_vctrAddrReg), Op(_op), 
    NRow(_nRow), NCol(_nCol), Mode(_mode)
{}

MatRowDotInsn::MatRowDotInsn(
        regNo_t _outAddrReg, 
        regNo_t _in1AddrReg, regNo_t _in1RowStrideReg,
        regNo_t _in2AddrReg, regNo_t _in2RowStrideReg,
        uint32_t _nRow, uint32_t _nCol, 
        bool _toAccBuf, bool _doAcc, ModeCode _mode) :
    OutAddrReg(_outAddrReg), 
    In1AddrReg(_in1AddrReg), In1RowStrideReg(_in1RowStrideReg),
    In2AddrReg(_in2AddrReg), In2RowStrideReg(_in2RowStrideReg),
    NRow(_nRow), NCol(_nCol), toAccBuf(_toAccBuf), doAcc(_doAcc), Mode(_mode)
{}

VctrSclrInsn::VctrSclrInsn(
        regNo_t _outAddrReg, 
        regNo_t _vctrAddrReg, regNo_t _sclrAddrReg,
        uint32_t _size, VctrSclrOp _op, ModeCode _mode) :
    OutAddrReg(_outAddrReg), 
    VctrAddrReg(_vctrAddrReg), SclrAddrReg(_sclrAddrReg),
    Op(_op), Size(_size), Mode(_mode)
{}

CopyAcc2BufInsn::CopyAcc2BufInsn(
        regNo_t _dstAddrReg,
        regNo_t _srcAddrReg,
        regNo_t _nUnitReg, 
        ModeCode _mode) :
    DstAddrReg(_dstAddrReg),
    SrcAddrReg(_srcAddrReg),
    SizeReg(_nUnitReg), Mode(_mode)
{}

DMA2BufferInsn::DMA2BufferInsn(
        DMADIR _dir, 
        regNo_t _hostPhyAddrReg, regNo_t _hostOffsetReg, 
        regNo_t _bufAddrReg, 
        regNo_t _sizeReg) :
    Dir(_dir),
    HostPhyAddrReg(_hostPhyAddrReg),
    HostOffsetReg(_hostOffsetReg),
    BufAddrReg(_bufAddrReg),
    SizeReg(_sizeReg)
{}

}  // end namespace nnpu