/*
runtime function declarations of NNPU simulator
*/

#ifndef NNPU_RUNTIME_H
#define NNPU_RUNTIME_H

#include <cstdint>
#include <vector>
#include "driver.h"
#include "insn.h"

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

void NNPU_VEXP(uint32_t vctr_out_addr, uint32_t vctr_in_addr, uint32_t len);

void NNPU_DMALoad(void *dram_buf_addr, uint32_t dram_buf_offset,
                  nnpu_dram_addr_t dst_phy_addr, uint32_t dst_phy_offset,
                  uint32_t size);

void NNPU_DMAStore(void *dram_buf_addr, uint32_t dram_buf_offset,
                  nnpu_dram_addr_t src_phy_addr, uint32_t src_phy_offset,
                  uint32_t size);

void NNPU_ScratchpadLoad(nnpu_dram_addr_t src_phy_addr, uint32_t src_offset,
                        nnpu_buf_addr_t dst_phy_addr, uint32_t dst_offset,
                        uint32_t size);

void NNPU_ScratchpadStore(nnpu_dram_addr_t dst_phy_addr, uint32_t dst_offset,
                        nnpu_buf_addr_t src_phy_addr, uint32_t src_offset,
                        uint32_t size);

void NNPUSynchronize(uint32_t timeout);

#ifdef __cplusplus
}
#endif

#endif