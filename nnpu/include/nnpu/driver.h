/*
driver function declarations of nnpu simulator
*/

#ifndef NNPU_DRIVER_H
#define NNPU_DRIVER_H

#include <cstdint>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <map>
#include <nnpusim/typedef.h>
#include <nnpusim/micro_code.h>

namespace nnpu
{
    struct NNPUInsn;  // forward declare NNPU insn for simulation.
}  // end namespace nnpu

#ifdef __cplusplus
extern "C"{
#endif

using nnpu_phy_addr_t = uint32_t;

using std::size_t;

void NNPUInvalidateCache(uint32_t ptr, size_t size);

void NNPUFlushCache(uint32_t ptr, size_t size);

void* NNPUMemAlloc(size_t size, int cached);

uint32_t NNPUMemGetPhyAddr(void *buf);

void NNPUMemFree(void *buf);

void NNPU_Run(const std::vector<nnpu::NNPUInsn> &insns, std::vector<nnpu::micro_kernel_t> &micro_kernels);

/*!
 * \brief relate device[id] with current thread.
*/
void NNPUSetDevice(int id);

#ifdef __cplusplus
}
#endif

namespace nnpu
{

enum class DevType
{
    S0, S1, SC
};

/*
base class for simulators
*/
class Simulator
{
public:

    const DevType devType;

    /*!
    * \brief run instructions on the simultor, pure virtual method.
    * \param insns: instructions to run.
    * \param pc: program counter where execution starts.
    */
    virtual void Run(const std::vector<NNPUInsn> &insns, std::size_t pc,
                     const std::vector<micro_kernel_t> &micro_kernels) = 0;

    virtual ~Simulator() {}

    /*!
     * \brief set register value,
     *        simulator implementations have to implement this function.
     * \param regNo: register number to be set.
     * \param regVal: the value to set.
     */
    virtual void WriteRegister(regNo_t regNo, reg_t regVal) = 0;

    /*!
     * \brief writer to scalar memory.
     *        used to pass arguments before running function on device.
     * \param addr: the address to write.
     * \param value: the value to be written.
     */
    virtual void WriteSclrMem(std::size_t addr, reg_t value) = 0;

    /*!
     * \brief get the scalar memory size.
     *        simulator implementions have to implement this function.
     * \return scalar memory size.
    */
    virtual std::size_t GetSclrMemSize() const = 0;

    /**!
     * \brief get the core extent, ie, core numbers.
     *        default implementation returns 1, multi-core simulator should return corresponding value.
    */
    virtual unsigned GetCoreExtent() const;

    // some static members for device setting and finding.
    /*!
    * \brief get a thread local simulator instance 
    */
    static std::shared_ptr<Simulator> ThreadLocal();

    static void SetThreadLocal(std::shared_ptr<Simulator> sim);

    /*! 
    * \brief default simulator type
    */
    static DevType DefaultType;

    static void SetDevice(int id, std::shared_ptr<Simulator> device);

    static std::shared_ptr<Simulator> GetDevice(int id);

protected:
    /*!
     * \brief base simulator default constructor
     * \param type: the device type(or just indicator)
     */
    Simulator(DevType type) : devType(type) {}

private:
    static std::unordered_map<int, std::shared_ptr<Simulator>> devices;
};

enum class NNPUReg { Zero = 0, SP = 1, FP = 2, CoreIdx = 3 };
inline regNo_t getRegNo(NNPUReg reg) {
    return static_cast<regNo_t>(reg);
}

/*!
* class copied from tvm.vta, used for managing buffer on host.
* \brief DRAM memory manager
*  Implements simple paging to allow physical address translation.
*/
class DRAM
{
  public:
    /*!
     * \brief Get virtual address given physical address.
     * \param phy_addr The simulator phyiscal address.
     * \return The true virtual address;
     */
    void *GetAddr(uint64_t phy_addr);
    
    /*!
     * \brief Get physical address
     * \param buf The virtual address.
     * \return The true physical address;
     */
    nnpu_phy_addr_t GetPhyAddr(void *buf);

    /*!
     * \brief Allocate memory from manager
     * \param size The size of memory
     * \return The virtual address
     */
    void *Alloc(size_t size);

    /*!
     * \brief Free the memory.
     * \param size The size of memory
     * \return The virtual address
     */
    void Free(void *data);

    static DRAM *Global();

  private:
    // The bits in page table
    static constexpr nnpu_phy_addr_t kPageBits = 12;
    // page size, also the maximum allocable size 16 K
    static constexpr nnpu_phy_addr_t kPageSize = 1 << kPageBits;
    /*! \brief A page in the DRAM */
    struct Page
    {
        /*! \brief Data Type */
        using DType = typename std::aligned_storage<kPageSize, 256>::type;
        /*! \brief Start location in page table */
        size_t ptable_begin;
        /*! \brief The total number of pages */
        size_t num_pages;
        /*! \brief Data */
        DType *data{nullptr};
        // construct a new page
        explicit Page(size_t ptable_begin, size_t num_pages)
            : ptable_begin(ptable_begin), num_pages(num_pages)
        {
            data = new DType[num_pages];
        }
        ~Page()
        {
            delete[] data;
        }
    };
    // Internal lock
    std::mutex mutex_;
    // Physical address -> page
    std::vector<Page *> ptable_;
    // virtual addres -> page
    std::unordered_map<void *, std::unique_ptr<Page>> pmap_;
    // Free map
    std::multimap<size_t, Page *> free_map_;
};

} // namespace nnpu

#endif