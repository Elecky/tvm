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

void NNPU_VEXP(uint32_t vctr_out_addr, uint32_t vctr_in_addr, uint32_t len)
{
    LOG(INFO) << "VEXP " << vctr_out_addr << ", " << vctr_in_addr << ", " << len << std::endl;

    nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

    nnpu::LiInsn li(0, vctr_out_addr);
    queue->EmplaceBack(li);

}

void NNPU_DMALoad(void *dram_buf_addr, uint32_t dram_buf_offset,
                  nnpu_dram_addr_t dst_phy_addr, uint32_t dst_phy_offset,
                  uint32_t size)
{
    LOG(INFO) << "DMALoad " 
              << dram_buf_addr << "(" << dram_buf_offset << "), " 
              << dst_phy_addr << "(" << dst_phy_offset << ")" << ", "
              << size << std::endl;
}

void NNPU_DMAStore(void *dram_buf_addr, uint32_t dram_buf_offset,
                  nnpu_dram_addr_t src_phy_addr, uint32_t src_phy_offset,
                  uint32_t size)
{
    LOG(INFO) << "DMAStore " 
              << dram_buf_addr << "(" << dram_buf_offset << "), " 
              << src_phy_addr << "(" << src_phy_offset << ")" << ", "
              << size << std::endl;
}

void NNPU_ScratchpadLoad(nnpu_dram_addr_t src_phy_addr, uint32_t src_offset,
                        nnpu_buf_addr_t dst_phy_addr, uint32_t dst_offset,
                        uint32_t size)
{
    LOG(INFO) << "ScratchpadLoad "
              << src_phy_addr << "(" << src_offset << "), "
              << dst_phy_addr << "(" << dst_offset << "), "
              << size << std::endl;
}

void NNPU_ScratchpadStore(nnpu_dram_addr_t dst_phy_addr, uint32_t dst_offset,
                        nnpu_buf_addr_t src_phy_addr, uint32_t src_offset,
                        uint32_t size)
{
    LOG(INFO) << "ScratchpadStore "
              << dst_phy_addr << "(" << dst_offset << "), "
              << src_phy_addr << "(" << src_offset << "), "
              << size << std::endl;
}

void NNPUSynchronize(uint32_t timeout)
{
    LOG(INFO) << "Sync" << std::endl;

    LOG(INFO) << nnpu::InsnQueue::ThreadLocal()->GetInsns().size() << " instructions to run";
}