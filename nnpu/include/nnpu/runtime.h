/*
runtime function declarations of NNPU simulator
*/

#ifndef NNPU_RUNTIME_H
#define NNPU_RUNTIME_H

#include <cstdint>
#include <vector>
#include <nnpu/driver.h>
#include <nnpu/insn.h>

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

void NNPU_AssembleAndRun(std::string asm_code, 
                         std::string func_name,
                         int coproc_scope, 
                         std::vector<int32_t> args);

#ifdef __cplusplus
}
#endif

#endif