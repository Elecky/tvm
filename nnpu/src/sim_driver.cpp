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
#include <yaml-cpp/yaml.h>
#include <nnpusim/S0Simulator.h>
#include <tvm/runtime/registry.h>

namespace nnpu
{

// DRAM member method implemention
/*!
* \brief Get virtual address given physical address.
* \param phy_addr The simulator phyiscal address.
* \return The true virtual address;
*/
void *DRAM::GetAddr(uint64_t phy_addr)
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
nnpu_phy_addr_t DRAM::GetPhyAddr(void *buf)
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
void *DRAM::Alloc(size_t size)
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
void DRAM::Free(void *data)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (pmap_.size() == 0)
        return;
    auto it = pmap_.find(data);
    CHECK(it != pmap_.end());
    Page *p = it->second.get();
    free_map_.insert(std::make_pair(p->num_pages, p));
}

DRAM *DRAM::Global()
{
    static DRAM inst;
    return &inst;
}

// set default simulator type as S0
DevType Simulator::DefaultType = DevType::S0;

/*!
* use dmlc Thread Local to achieve thread local instance.
*/
std::shared_ptr<Simulator> Simulator::ThreadLocal()
{
    std::shared_ptr<Simulator> ptr = *(dmlc::ThreadLocalStore<std::shared_ptr<Simulator>>::Get());

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

/*!
* \brief construct a nnpu simulator based on given configuration
* \param:
*    type: simulator type
*    cfg: a yaml configure node
*/
std::shared_ptr<nnpu::Simulator> NNPUDevAlloc(nnpu::DevType type, YAML::Node cfg)
{
    using Type = nnpu::DevType;
    switch (type)
    {
    case Type::S0:
        return std::make_shared<nnpu::S0Simulator>(cfg);
        break;

    default:
        return nullptr;
    }
}

// Register global function to set simulator device
TVM_REGISTER_GLOBAL("nnpu.set_dev")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
        // there should be 2 arguments
        if (args.num_args != 2)
        {
            *rv = std::string("expect 2 arguments");
            return;
        }

        using Type = nnpu::DevType;
        Type devType;
        std::string t_str = args[0];

        if (t_str == std::string("S0"))
        {
            devType = Type::S0;
        }
        else if (t_str == std::string("S1"))
        {
            devType = Type::S1;
        }
        else
        {
            *rv = std::string("unhandled simulator type");
            return;
        }

        std::string cfg_path = args[1];
        YAML::Node cfg = YAML::LoadFile(cfg_path);

        nnpu::Simulator::ThreadLocal() = NNPUDevAlloc(devType, cfg);
    });