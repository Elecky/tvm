/*
this is the base simulator driver of NNPU,
some common functionalities are implemented here.
also declares the base simulator interface.
*/

#include <nnpu/driver.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <mutex>
#include <unordered_map>
#include <map>

namespace nnpu
{

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
    void *GetAddr(uint64_t phy_addr)
    {
        CHECK_NE(phy_addr, 0)
            << "trying to get address that is nullptr";
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t loc = (phy_addr >> kPageBits) - 1;
        CHECK_LT(loc, ptable_.size())
            << "phy_addr=" << phy_addr;
        Page *p = ptable_[loc];
        CHECK(p != nullptr);
        size_t offset = (loc - p->ptable_begin) << kPageBits;
        offset += phy_addr & (kPageSize - 1);
        return reinterpret_cast<char *>(p->data) + offset;
    }
    /*!
     * \brief Get physical address
     * \param buf The virtual address.
     * \return The true physical address;
     */
    nnpu_phy_addr_t GetPhyAddr(void *buf)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pmap_.find(buf);
        CHECK(it != pmap_.end());
        Page *p = it->second.get();
        return (p->ptable_begin + 1) << kPageBits;
    }
    /*!
     * \brief Allocate memory from manager
     * \param size The size of memory
     * \return The virtual address
     */
    void *Alloc(size_t size)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t npage = (size + kPageSize - 1) / kPageSize;
        auto it = free_map_.lower_bound(npage);
        if (it != free_map_.end())
        {
            Page *p = it->second;
            free_map_.erase(it);
            return p->data;
        }
        size_t start = ptable_.size();
        std::unique_ptr<Page> p(new Page(start, npage));
        // insert page entry
        ptable_.resize(start + npage, p.get());
        void *data = p->data;
        pmap_[data] = std::move(p);
        return data;
    }

    /*!
     * \brief Free the memory.
     * \param size The size of memory
     * \return The virtual address
     */
    void Free(void *data)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pmap_.size() == 0)
            return;
        auto it = pmap_.find(data);
        CHECK(it != pmap_.end());
        Page *p = it->second.get();
        free_map_.insert(std::make_pair(p->num_pages, p));
    }

    static DRAM *Global()
    {
        static DRAM inst;
        return &inst;
    }

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

// set default simulator type as S0
DevType Simulator::DefaultType = DevType::S0;

/*
*  use dmlc Thread Local to achieve thread local instance.
*/
std::shared_ptr<Simulator> Simulator::ThreadLocal()
{
    std::shared_ptr<Simulator> ptr = *(dmlc::ThreadLocalStore<std::shared_ptr<Simulator>>::Get());
    if (ptr == nullptr)
    {
        // maybe allocate a simulator here?

    }

    return ptr;
}

} // namespace nnpu

void NNPUInvalidateCache(uint32_t ptr, size_t size)
{
}

void NNPUFlushCache(uint32_t ptr, size_t size)
{
}

void *NNPUMemAlloc(size_t size, int cached)
{
    return nnpu::DRAM::Global()->Alloc(size);
}

uint32_t NNPUMemGetPhyAddr(void *buf)
{
    return nnpu::DRAM::Global()->GetPhyAddr(buf);
}

void NNPUMemFree(void *buf)
{
    return nnpu::DRAM::Global()->Free(buf);
}