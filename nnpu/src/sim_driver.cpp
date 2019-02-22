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
#include <tvm/runtime/registry.h>
#include <nnpu/insn.h>

using std::shared_ptr;

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

std::unordered_map<int, std::shared_ptr<Simulator>> Simulator::devices;

/*!
* use dmlc Thread Local to achieve thread local instance.
*/
shared_ptr<Simulator> Simulator::ThreadLocal()
{
    return *(dmlc::ThreadLocalStore<shared_ptr<Simulator>>::Get());
}

void Simulator::SetThreadLocal(std::shared_ptr<Simulator> sim)
{
    *(dmlc::ThreadLocalStore<shared_ptr<Simulator>>::Get()) = std::move(sim);
}

std::shared_ptr<Simulator> Simulator::GetDevice(int id)
{
    auto it = devices.find(id);
    if (it != devices.end())
    {
        return it->second;
    }
    else
    {
        return nullptr;
    }
}

void Simulator::SetDevice(int id, shared_ptr<Simulator> device)
{
    devices[id] = std::move(device);
}

} // end namespace nnpu

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

void NNPU_Run(const std::vector<nnpu::NNPUInsn> &insns)
{
    //LOG(INFO) << "pointer to simulator is: " << nnpu::Simulator::ThreadLocal();
    nnpu::Simulator::ThreadLocal()->Run(insns);
}

void NNPUSetDevice(int id)
{
    nnpu::Simulator::SetThreadLocal(nnpu::Simulator::GetDevice(id));
}

namespace nnpu
{
// simulator creaters
std::shared_ptr<nnpu::Simulator> createS0Simulator(YAML::Node cfg);

std::shared_ptr<nnpu::Simulator> createS1Simulator(YAML::Node cfg);
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
        return nnpu::createS0Simulator(cfg);

    case Type::S1:
        return nnpu::createS1Simulator(cfg);

    default:
        return nullptr;
    }
}

// Register global function to set simulator device
TVM_REGISTER_GLOBAL("nnpu.set_dev")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
        // there should be 2 arguments
        if (args.num_args != 3)
        {
            *rv = std::string("expect 3 arguments");
            return;
        }

        using Type = nnpu::DevType;
        Type devType;
        std::string t_str = args[1];

        static const std::string s0("S0"), s1("S1");

        if (t_str == s0)
        {
            devType = Type::S0;
        }
        else if (t_str == s1)
        {
            devType = Type::S1;
        }
        else
        {
            *rv = std::string("unhandled simulator type");
            return;
        }

        std::string cfg_path = args[2];
        YAML::Node cfg = YAML::LoadFile(cfg_path);

        int device_id = args[0];

        nnpu::Simulator::SetDevice(device_id, NNPUDevAlloc(devType, cfg));
    });