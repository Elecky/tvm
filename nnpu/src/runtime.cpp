/*
NNPU runtime
*/

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <nnpu/driver.h>
#include <nnpu/insn.h>
#include <nnpu/runtime.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <vector>
#include <sstream>

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

using std::vector;
using std::string;

string Trim(string str)
{
    auto first = str.find_first_not_of(' ');
    if (first == string::npos)
    {
        return string();
    }
    else
    {
        string res = str.substr(first);
        auto last = res.find_last_not_of(' ');
        res.resize(last + 1);
        return res;
    }
}

class NNPUAssembler
{
public:
    NNPUAssembler() = default;

    /*
     * \brief assemble the asm code, creating instructions.
     * \param asm_str the assembly code string.
     */
    void Assemble(string asm_str);

private:
    vector<NNPUInsn> insns;
};

void NNPUAssembler::Assemble(string asm_str)
{
    std::stringstream ss(asm_str);
    constexpr std::size_t bufferSize = 1024;
    std::unique_ptr<char[]> buffer(new char[bufferSize]);

    while (ss.getline(buffer.get(), bufferSize))
    {
        string raw = string(buffer.get());
        string line = Trim(raw);
    }
}

}  // end namespace nnpu

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

using std::vector;
using std::string;

static bool DumpInsn = true;
using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;

static TVM_ATTRIBUTE_UNUSED auto &__register_set_dump_ =
    ::tvm::runtime::Registry::Register("nnpu.set_dump", true)
        .set_body([](TVMArgs args, TVMRetValue *rv) {
            if (args.size() >= 1)
                DumpInsn = static_cast<bool>(args[0]);
        });

static uint32_t NNPU_Handle2PhyAddr(void *handle)
{
    auto buffer = nnpu::DataBuffer::FromHandle(handle);
    return buffer->phy_addr();
}

static TVM_ATTRIBUTE_UNUSED auto &__register_handleTophyAddr_ =
    ::tvm::runtime::Registry::Register("nnpu.handleToPhyAddr", true)
        .set_body([](TVMArgs args, TVMRetValue *rv) {
            CHECK(args.num_args >= 1) << ", ecpected one argument";
            CHECK(rv != nullptr) << ", empty return address";

            (*rv) = static_cast<int64_t>(
                        NNPU_Handle2PhyAddr(static_cast<void*>(args[0])));
        });

extern "C" void NNPU_AssembleAndRun(
                    string asm_code, 
                    string func_name, 
                    int coproc_scope,
                    std::vector<int32_t> args)
{
    auto &os = LOG(INFO);
    os << "NNPU runtime function: NNPU_AssembleAndRun";
    os << "\n call args:\n  [";
    for (auto it : args)
    {
        os << it << ", ";
    }
    os << "]\n coproc scope = " << coproc_scope;
    os << "\n calling function [" << func_name;
    os << "] in asm code: \n";
    os << asm_code;
}

static TVM_ATTRIBUTE_UNUSED auto &__register_run_ =
    ::tvm::runtime::Registry::Register("nnpu.assemble_and_run", true)
        .set_body([](TVMArgs args, TVMRetValue *rv) {
            CHECK_GE(args.num_args, 3) << ", ecpected at least 3 arguments";
            CHECK_EQ(args.type_codes[0], kStr) 
                << ", expecting 1st argument to be assembly code [string]";
            CHECK_EQ(args.type_codes[1], kStr)
                << ", expecting 2nd argument to be function name [string]";
            CHECK_EQ(args.type_codes[2], kDLInt)
                << ", expecting 3rd argument to be coproc scope [int]";

            std::vector<int32_t> dev_args;  // arguments to be passed to device function.
            for (int i = 3; i < args.num_args; ++i)
            {
                CHECK_EQ(args.type_codes[i], kDLInt)
                    << ", only int type arguments can be passed to NNPU device";
                dev_args.push_back(static_cast<int32_t>(args[i]));
            }

            NNPU_AssembleAndRun(args[0].operator std::__cxx11::string(), 
                                args[1].operator std::__cxx11::string(),
                                static_cast<int>(args[2]),
                                dev_args);
        });