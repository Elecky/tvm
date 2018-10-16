/*
NNPU runtime
*/

#include <cstdlib>
#include <cstring>
#include <nnpu/driver.h>
#include <nnpu/insn.h>
#include <nnpu/runtime.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

static const bool kBufferCoherent = false;

namespace nnpu
{

/*!
 * ack: code from tvm.vta
 * \brief Data buffer represents data on CMA.
 */
struct DataBuffer
{
    /*! \return Virtual address of the data. */
    void *virt_addr() const
    {
        return data_;
    }
    /*! \return Physical address of the data. */
    uint32_t phy_addr() const
    {
        return phy_addr_;
    }
    /*!
    * \brief Invalidate the cache of given location in data buffer.
    * \param offset The offset to the data.
    * \param size The size of the data.
    */
    void InvalidateCache(size_t offset, size_t size)
    {
        if (!kBufferCoherent)
        {
            NNPUInvalidateCache(phy_addr_ + offset, size);
        }
    }
    /*!
    * \brief Invalidate the cache of certain location in data buffer.
    * \param offset The offset to the data.
    * \param size The size of the data.
    */
    void FlushCache(size_t offset, size_t size)
    {
        if (!kBufferCoherent)
        {
            NNPUFlushCache(phy_addr_ + offset, size);
        }
    }
    /*!
    * \brief Allocate a buffer of a given size.
    * \param size The size of the buffer.
    */
    static DataBuffer *Alloc(size_t size)
    {
        void *data = NNPUMemAlloc(size, 1);
        CHECK(data != nullptr);
        DataBuffer *buffer = new DataBuffer();
        buffer->data_ = data;
        buffer->phy_addr_ = NNPUMemGetPhyAddr(data);
        return buffer;
    }
    /*!
    * \brief Free the data buffer.
    * \param buffer The buffer to be freed.
    */
    static void Free(DataBuffer *buffer)
    {
        NNPUMemFree(buffer->data_);
        delete buffer;
    }
    /*!
    * \brief Create data buffer header from buffer ptr.
    * \param buffer The buffer pointer.
    * \return The corresponding data buffer header.
    */
    static DataBuffer *FromHandle(const void *buffer)
    {
        return const_cast<DataBuffer *>(
            reinterpret_cast<const DataBuffer *>(buffer));
    }

  private:
    /*! \brief The internal data. */
    void *data_;
    /*! \brief The physical address of the buffer, excluding header. */
    uint32_t phy_addr_;
};


/* implemention of nnpu::InsnQueue*/
using std::vector;

// the instruction queue, stores instruction when doing JIT compilation.
class InsnQueue
{
public:
    /* 
    *  InsnQueue default constructor.
    *  currently does nothing
    */
    InsnQueue()
    {}

    /* get a thread local InsnQueue object pointer
    */
    static InsnQueue* ThreadLocal()
    {
        // use dmlc ThreadLocal library to achieve thread local
        return dmlc::ThreadLocalStore<InsnQueue>::Get();
    }

    template <typename T>
    void PushInsn(T&& arg)
    {
        insnQueue.push_back(arg);
    }

    template <typename ... Ts>
    void EmplaceBack(Ts&& ... args)
    {
        insnQueue.emplace_back(args...);
    }

    /* reset the instruction queue, which clears all instructions alread in it.
    */
    void Reset()
    {
        // simply clears the underline vector.
        insnQueue.clear();
    }

    vector<NNPUInsn>& GetInsns()
    {
        return insnQueue;
    }

    /*!
    * \brief dump all instructions(string repr) into ostream.
    * \param os, the ostream into which to dump the instructions.
    */
    void Dump(std::ostream& os)
    {
        InsnDumper dumper;

        for (auto& insn : insnQueue)
        {
            insn.Call(dumper, os);
            os << std::endl;
        }
    }

private:
    vector<NNPUInsn> insnQueue;
};

} // namespace nnpu

// the following 3 functions are from tvm.vta, used for managing driver buffers.
void *NNPUBufferAlloc(size_t size)
{
    return nnpu::DataBuffer::Alloc(size);
}

void NNPUBufferFree(void *buffer)
{
    return nnpu::DataBuffer::Free(nnpu::DataBuffer::FromHandle(buffer));
}

void NNPUBufferCopy(const void *from,
                    size_t from_offset,
                    void *to,
                    size_t to_offset,
                    size_t size,
                    int kind_mask)
{
    nnpu::DataBuffer *from_buffer = nullptr;
    nnpu::DataBuffer *to_buffer = nullptr;

    if (kind_mask & 2)  // source is accelerator
    {
        from_buffer = nnpu::DataBuffer::FromHandle(from);
        from = from_buffer->virt_addr();
    }
    if (kind_mask & 1)  // destination is accelerator
    {
        to_buffer = nnpu::DataBuffer::FromHandle(to);
        to = to_buffer->virt_addr();
    }
    if (from_buffer)
    {
        from_buffer->InvalidateCache(from_offset, size);
    }

    memcpy(static_cast<char *>(to) + to_offset,
           static_cast<const char *>(from) + from_offset,
           size);
    if (to_buffer)
    {
        to_buffer->FlushCache(to_offset, size);
    }
}

void NNPU_VEXP(uint32_t vctr_out_addr, uint32_t vctr_in_addr, uint32_t len, uint32_t mode)
{
    LOG(INFO) << "VEXP " << vctr_out_addr << ", " << vctr_in_addr << ", " << len << std::endl;

    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // load vector out address into $0
    nnpu::LiInsn li1(0, vctr_out_addr);
    queue->EmplaceBack(li1);
    // vector in address into $1
    nnpu::LiInsn li2(1, vctr_in_addr);
    queue->EmplaceBack(li2);
    // element count into $2
    nnpu::LiInsn li3(2, len);
    queue->EmplaceBack(li3);

    // create a vctr exp instruction: VEXP $0, $1, $2
    nnpu::VctrUnaryInsn exp(nnpu::VctrUnaryOp::Exp, 0, 1, 2, nnpu::ModeFromInt(mode));
    queue->EmplaceBack(exp);
}
void NNPU_VLOG(uint32_t vctr_out_addr, uint32_t vctr_in_addr, uint32_t len, uint32_t mode)
{
    LOG(INFO) << "VLOG " << vctr_out_addr << ", " << vctr_in_addr << ", " << len << std::endl;

    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // load vector out address into $0
    nnpu::LiInsn li1(0, vctr_out_addr);
    queue->EmplaceBack(li1);
    // vector in address into $1
    nnpu::LiInsn li2(1, vctr_in_addr);
    queue->EmplaceBack(li2);
    // element count into $2
    nnpu::LiInsn li3(2, len);
    queue->EmplaceBack(li3);

    // create a vctr log instruction: VLOG $0, $1, $2
    nnpu::VctrUnaryInsn log(nnpu::VctrUnaryOp::Log, 0, 1, 2, nnpu::ModeFromInt(mode));
    queue->EmplaceBack(log);
}

void NNPU_DMALoad(void *host_buf_addr, uint32_t host_buf_offset,
                  nnpu_dram_addr_t dst_phy_addr, uint32_t dst_phy_offset,
                  uint32_t size)
{
    LOG(INFO) << "DMALoad " 
              << host_buf_addr << "(" << host_buf_offset << "), " 
              << dst_phy_addr << "(" << dst_phy_offset << ")" << ", "
              << size << std::endl;

    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // load host buffer physical address and offset into $0 and $1
    nnpu::DataBuffer *buffer = nnpu::DataBuffer::FromHandle(host_buf_addr);
    Li liAddr1(0, buffer->phy_addr());
    queue->EmplaceBack(liAddr1);
    Li liOff1(1, host_buf_offset);
    queue->EmplaceBack(liOff1);

    // load dram address into $2
    Li liAddr2(2, dst_phy_addr + dst_phy_offset);
    queue->EmplaceBack(liAddr2);

    // load copy size in byte into $3
    Li liSize(3, size);
    queue->EmplaceBack(liSize);
    // create a DMA load instruction: DMALoad $0, $1, $2, $3
    nnpu::DMACopyInsn copyInsn(nnpu::DMADIR::HtoD, 0, 1, 2, 3);
    queue->EmplaceBack(copyInsn);
}

void NNPU_DMAStore(void *host_buf_addr, uint32_t host_buf_offset,
                  nnpu_dram_addr_t src_phy_addr, uint32_t src_phy_offset,
                  uint32_t size)
{
    LOG(INFO) << "DMAStore " 
              << host_buf_addr << "(" << host_buf_offset << "), " 
              << src_phy_addr << "(" << src_phy_offset << ")" << ", "
              << size << std::endl;
    
    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // load host buffer physical address and offset into $0 and $1
    nnpu::DataBuffer *buffer = nnpu::DataBuffer::FromHandle(host_buf_addr);
    Li liAddr1(0, buffer->phy_addr());
    queue->EmplaceBack(liAddr1);
    Li liOff1(1, host_buf_offset);
    queue->EmplaceBack(liOff1);

    // load dram address into $2
    Li liAddr2(2, src_phy_addr + src_phy_offset);
    queue->EmplaceBack(liAddr2);

    // load copy size in byte into $3
    Li liSize(3, size);
    queue->EmplaceBack(liSize);
    // create a DMA load instruction: DMALoad $0, $1, $2, $3
    nnpu::DMACopyInsn copyInsn(nnpu::DMADIR::DtoH, 0, 1, 2, 3);
    queue->EmplaceBack(copyInsn);
}

void NNPU_ScratchpadLoad(nnpu_dram_addr_t src_phy_addr, uint32_t src_offset,
                        nnpu_buf_addr_t dst_phy_addr, uint32_t dst_offset,
                        uint32_t size)
{
    LOG(INFO) << "ScratchpadLoad "
              << src_phy_addr << "(" << src_offset << "), "
              << dst_phy_addr << "(" << dst_offset << "), "
              << size << std::endl;

    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // load dram address into $0
    Li liAddr1(0, src_phy_addr + src_offset);
    queue->EmplaceBack(liAddr1);

    // load scratchpad address into $1
    Li liAddr2(1, dst_phy_addr + dst_offset);
    queue->EmplaceBack(liAddr2);

    // $2 <- size
    Li liSize(2, size);
    queue->EmplaceBack(liSize);

    // create a scratchpad load instruction: load.b $0, $1, $2
    nnpu::BufferLSInsn load(nnpu::LSDIR::Load, 0, 1, 2);
    queue->EmplaceBack(load);
}

void NNPU_ScratchpadStore(nnpu_dram_addr_t dst_phy_addr, uint32_t dst_offset,
                        nnpu_buf_addr_t src_phy_addr, uint32_t src_offset,
                        uint32_t size)
{
    LOG(INFO) << "ScratchpadStore "
              << dst_phy_addr << "(" << dst_offset << "), "
              << src_phy_addr << "(" << src_offset << "), "
              << size << std::endl;
    
    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // load dram address into $0
    Li liAddr1(0, dst_phy_addr + dst_offset);
    queue->EmplaceBack(liAddr1);

    // load scratchpad address into $1
    Li liAddr2(1, src_phy_addr + src_offset);
    queue->EmplaceBack(liAddr2);

    // $2 <- size
    Li liSize(2, size);
    queue->EmplaceBack(liSize);

    // create a scratchpad load instruction: load.b $0, $1, $2
    nnpu::BufferLSInsn load(nnpu::LSDIR::Store, 0, 1, 2);
    queue->EmplaceBack(load);
}

void NNPU_Gemm(uint32_t nRowOut, uint32_t factor, uint32_t nColOut, 
             uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t mode)
{
    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // load 3 addresses
    Li li1(0, outAddr);
    queue->EmplaceBack(li1);
    
    Li li2(1, in1Addr);
    queue->EmplaceBack(li2);

    Li li3(2, in2Addr);
    queue->EmplaceBack(li3);

    // create a gemm instruction
    nnpu::GemmInsn gemm(nRowOut, factor, nColOut, 0, 1, 2, nnpu::ModeFromInt(mode));
    queue->EmplaceBack(gemm);
}

void NNPUSynchronize(uint32_t timeout)
{
    LOG(INFO) << "Sync" << std::endl;

    //LOG(INFO) << "instructions to run: " << std::endl;

    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    nnpu::StallInsn stall;
    queue->EmplaceBack(stall);

    nnpu::InsnQueue::ThreadLocal()->Dump(LOG(INFO) << std::endl);

    NNPU_Run(queue->GetInsns());
}

void NNPU_VctrBinary(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, 
                    uint32_t size, uint32_t mode, nnpu::VctrBinaryOp op)
{
    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // assign 3 addresses
    Li li1(0, outAddr);
    queue->EmplaceBack(li1);
    Li li2(1, in1Addr);
    queue->EmplaceBack(li2);
    Li li3(2, in2Addr);
    queue->EmplaceBack(li3);

    nnpu::VctrBinaryInsn insn(op, 0, 1, 2, size, nnpu::ModeFromInt(mode));
    queue->EmplaceBack(insn);
}
// xian fang zai zhe li yi xia...
void NNPU_VctrImm(uint32_t outAddr, uint32_t inAddr, double Imm, 
                    uint32_t size, uint32_t mode, nnpu::VctrImmOp op)
{
    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // assign 3 addresses
    Li li1(0, outAddr);
    queue->EmplaceBack(li1);
    Li li2(1, inAddr);
    queue->EmplaceBack(li2);
    nnpu::VctrImmInsn insn(op, 0, 1, Imm, size, nnpu::ModeFromInt(mode));
    queue->EmplaceBack(insn);
}
void NNPU_VAddI(uint32_t outAddr, uint32_t inAddr, const char* ImmS, uint32_t size, uint32_t mode)
{
    std::stringstream st;
    st<<ImmS;
    double Imm = 0;
    st>>Imm;
    NNPU_VctrImm(outAddr, inAddr, Imm, size, mode, nnpu::VctrImmOp::Add);
}

void NNPU_VSubI(uint32_t outAddr, uint32_t inAddr, const char* ImmS, uint32_t size, uint32_t mode)
{
    std::stringstream st;
    st<<ImmS;
    double Imm = 0;
    st>>Imm;
    NNPU_VctrImm(outAddr, inAddr, Imm, size, mode, nnpu::VctrImmOp::Sub);
}

void NNPU_ISubV(uint32_t outAddr, uint32_t inAddr, const char* ImmS, uint32_t size, uint32_t mode)
{
    std::stringstream st;
    st<<ImmS;
    double Imm = 0;
    st>>Imm;
    NNPU_VctrImm(outAddr, inAddr, Imm, size, mode, nnpu::VctrImmOp::RSub);
}

void NNPU_VMulI(uint32_t outAddr, uint32_t inAddr, const char* ImmS, uint32_t size, uint32_t mode)
{
    std::stringstream st;
    st<<ImmS;
    double Imm = 0;
    st>>Imm;
    NNPU_VctrImm(outAddr, inAddr, Imm, size, mode, nnpu::VctrImmOp::Mul);
}

void NNPU_VDivI(uint32_t outAddr, uint32_t inAddr, const char* ImmS, uint32_t size, uint32_t mode)
{
    std::stringstream st;
    st<<ImmS;
    double Imm = 0;
    st>>Imm;
    NNPU_VctrImm(outAddr, inAddr, Imm, size, mode, nnpu::VctrImmOp::Div);
}

void NNPU_VGTMI(uint32_t outAddr, uint32_t inAddr, const char* ImmS, uint32_t size, uint32_t mode)
{
    std::stringstream st;
    st<<ImmS;
    double Imm = 0;
    st>>Imm;
    NNPU_VctrImm(outAddr, inAddr, Imm, size, mode, nnpu::VctrImmOp::GTM);
}

void NNPU_VAddV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode)
{
    NNPU_VctrBinary(outAddr, in1Addr, in2Addr, size, mode, nnpu::VctrBinaryOp::Add);
}

void NNPU_VSubV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode)
{
    NNPU_VctrBinary(outAddr, in1Addr, in2Addr, size, mode, nnpu::VctrBinaryOp::Sub);
}

void NNPU_VMulV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode)
{
    NNPU_VctrBinary(outAddr, in1Addr, in2Addr, size, mode, nnpu::VctrBinaryOp::Mul);
}

void NNPU_VDivV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode)
{
    NNPU_VctrBinary(outAddr, in1Addr, in2Addr, size, mode, nnpu::VctrBinaryOp::Div);
}

void NNPU_VGTMV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode)
{
    NNPU_VctrBinary(outAddr, in1Addr, in2Addr, size, mode, nnpu::VctrBinaryOp::GTM);
}

void NNPU_VctrDotProd(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, 
                      uint32_t size, uint32_t mode)
{
    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // assign 3 addresses
    Li li1(0, outAddr);
    queue->EmplaceBack(li1);
    Li li2(1, in1Addr);
    queue->EmplaceBack(li2);
    Li li3(2, in2Addr);
    queue->EmplaceBack(li3);

    nnpu::VctrDotProdInsn insn(0, 1, 2, size, nnpu::ModeFromInt(mode));
    queue->EmplaceBack(insn);
}

void NNPU_VctrReduce(uint32_t outAddr, uint32_t inAddr, nnpu::ReduceOp op, uint32_t size, uint32_t mode)
{
    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // assign 3 addresses
    Li li1(0, outAddr);
    queue->EmplaceBack(li1);
    Li li2(1, inAddr);
    queue->EmplaceBack(li2);
    
    nnpu::VctrReduceInsn insn(0, 1, op, size, nnpu::ModeFromInt(mode));
    queue->EmplaceBack(insn);
}

void NNPU_VctrReduceSum(uint32_t outAddr, uint32_t inAddr, uint32_t size, uint32_t mode)
{
    NNPU_VctrReduce(outAddr, inAddr, nnpu::ReduceOp::Sum, size, mode);
}

void NNPU_VctrReduceMax(uint32_t outAddr, uint32_t inAddr, uint32_t size, uint32_t mode)
{
    NNPU_VctrReduce(outAddr, inAddr, nnpu::ReduceOp::Max, size, mode);
}

void NNPU_VctrReduceMin(uint32_t outAddr, uint32_t inAddr, uint32_t size, uint32_t mode)
{
    NNPU_VctrReduce(outAddr, inAddr, nnpu::ReduceOp::Min, size, mode);
}

void NNPU_MatBinary(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, 
                    nnpu::MatBinaryOp op, uint32_t Size, uint32_t mode)
{
    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // assign 3 addresses
    Li li1(0, outAddr);
    queue->EmplaceBack(li1);
    Li li2(1, in1Addr);
    queue->EmplaceBack(li2);
    Li li3(2, in2Addr);
    queue->EmplaceBack(li3);

    nnpu::MatBinaryInsn insn(0, 1, 2, op, Size, nnpu::ModeFromInt(mode));
    queue->EmplaceBack(insn);
}

void NNPU_MAddM(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t Size, uint32_t mode)
{
    NNPU_MatBinary(outAddr, in1Addr, in2Addr, nnpu::MatBinaryOp::Add, Size, mode);
}

void NNPU_MSubM(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t Size, uint32_t mode)
{
    NNPU_MatBinary(outAddr, in1Addr, in2Addr, nnpu::MatBinaryOp::Sub, Size, mode);
}

void NNPU_MMulM(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t Size, uint32_t mode)
{
    NNPU_MatBinary(outAddr, in1Addr, in2Addr, nnpu::MatBinaryOp::Mul, Size, mode);
}

void NNPU_MReduce(uint32_t outAddr, uint32_t inAddr, nnpu::ReduceOp op, 
                  uint32_t nRow, uint32_t nCol, uint32_t mode, bool isRow)
{
    using Li = nnpu::LiInsn;
    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    // assign 3 addresses
    Li li1(0, outAddr);
    queue->EmplaceBack(li1);
    Li li2(1, inAddr);
    queue->EmplaceBack(li2);

    if (isRow)
    {
        nnpu::MatReduceRowInsn insn(0, 1, op, nRow, nCol, nnpu::ModeFromInt(mode));
        queue->EmplaceBack(insn);
    }
    else
    {
        nnpu::MatReduceColInsn insn(0, 1, op, nRow, nCol, nnpu::ModeFromInt(mode));
        queue->EmplaceBack(insn);
    }
}

void NNPU_MReduceSumRow(uint32_t outAddr, uint32_t inAddr, uint32_t nRow, uint32_t nCol, 
                        uint32_t mode)
{
    NNPU_MReduce(outAddr, inAddr, nnpu::ReduceOp::Sum, nRow, nCol, mode, true);
}