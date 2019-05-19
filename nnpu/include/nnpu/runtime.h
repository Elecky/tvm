/*
runtime function declarations of NNPU simulator
*/

#ifndef NNPU_RUNTIME_H
#define NNPU_RUNTIME_H

#include <cstdint>
#include <vector>
#include <nnpu/driver.h>
#include <nnpu/insn.h>
#include <tvm/runtime/c_runtime_api.h>

using nnpu_dram_addr_t = uint32_t;
using nnpu_buf_addr_t = uint32_t;

#ifdef __cplusplus
extern "C" {
#endif

void* NNPUBufferAlloc(size_t size);

void NNPUBufferFree(void* buffer);

void NNPUBufferCopy(const void* from,
                    size_t from_offset,
                    void* to,
                    size_t to_offset,
                    size_t size,
                    int kind_mask);

/**!
 * \brief assemble and run on NNPU simulator
 * \param asm_code: the target assembly code of device.
 * \param func_name: which function to run.
 * \param core_extent: the number of cores this function should be launched on.
 * \param args: arguments passed to device function.
*/
void NNPU_AssembleAndRun(std::string asm_code, 
                         std::string func_name,
                        //  int coproc_scope, 
                         unsigned core_extent,
                         std::vector<int32_t> args);

TVM_DLL void *NNPUBufferCPUPtr(void *buffer);

#ifdef __cplusplus
}
#endif

#endif