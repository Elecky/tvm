/*
driver function declarations of nnpu simulator
*/

#ifndef NNPU_DRIVER_H
#define NNPU_DRIVER_H

#include <cstdint>
#include <memory>

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

#ifdef __cplusplus
}
#endif

namespace nnpu
{

enum class DevType
{
    S0, S1
};

/*
base class for simulators
*/
class Simulator
{
public:
    Simulator() = delete;

    const DevType devType;

    virtual ~Simulator() = 0;

    /* get a thread local simulator instance 
    */
    static std::shared_ptr<Simulator> ThreadLocal();

    /* default simulator type
    */
    static DevType DefaultType;
};

} // namespace nnpu

#endif