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

void NNPU_VEXP(uint32_t vctr_out_addr, uint32_t vctr_in_addr, uint32_t len, uint32_t mode);
void NNPU_VLOG(uint32_t vctr_out_addr, uint32_t vctr_in_addr, uint32_t len, uint32_t mode);
void NNPU_DMALoad(void *host_buf_addr, uint32_t host_buf_offset,
                  nnpu_dram_addr_t dst_phy_addr, uint32_t dst_phy_offset,
                  uint32_t size);

void NNPU_DMAStore(void *host_buf_addr, uint32_t host_buf_offset,
                  nnpu_dram_addr_t src_phy_addr, uint32_t src_phy_offset,
                  uint32_t size);

void NNPU_ScratchpadLoad(nnpu_dram_addr_t src_phy_addr, uint32_t src_offset,
                        nnpu_buf_addr_t dst_phy_addr, uint32_t dst_offset,
                        uint32_t size);

void NNPU_ScratchpadStore(nnpu_dram_addr_t dst_phy_addr, uint32_t dst_offset,
                        nnpu_buf_addr_t src_phy_addr, uint32_t src_offset,
                        uint32_t size);

void NNPU_Gemm(uint32_t nRowOut, uint32_t factor, uint32_t nColOut, 
               uint32_t outAddr, uint32_t outRowStride,
               uint32_t in1Addr, uint32_t in1RowStride,
               uint32_t in2Addr, uint32_t in2RowStride, uint32_t mode);

void NNPU_MAddI(uint32_t outAddr, uint32_t inAddr, const char* ImmS ,uint32_t nRow,uint32_t nCol, uint32_t mode);
void NNPU_MMulI(uint32_t outAddr, uint32_t inAddr, const char* ImmS ,uint32_t nRow,uint32_t nCol, uint32_t mode);
void NNPU_ISubM(uint32_t outAddr, uint32_t inAddr, const char* ImmS ,uint32_t nRow,uint32_t nCol, uint32_t mode);
void NNPU_VAddV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode);

void NNPU_VSubV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode);

void NNPU_VDivV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode);

void NNPU_VMulV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode);

void NNPU_VGTMV(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, uint32_t mode);

void NNPU_VAddI(uint32_t outAddr, uint32_t inAddr, const char* ImmS ,uint32_t size, uint32_t mode);

void NNPU_VMulI(uint32_t outAddr, uint32_t inAddr, const char* ImmS ,uint32_t size, uint32_t mode);

void NNPU_VSubI(uint32_t outAddr, uint32_t inAddr, const char* ImmS ,uint32_t size, uint32_t mode);

void NNPU_ISubV(uint32_t outAddr, uint32_t inAddr, const char* ImmS ,uint32_t size, uint32_t mode);

void NNPU_VDivI(uint32_t outAddr, uint32_t inAddr, const char* ImmS ,uint32_t size, uint32_t mode);

void NNPU_VGTMI(uint32_t outAddr, uint32_t inAddr, const char* ImmS ,uint32_t size, uint32_t mode);

void NNPU_IDivV(uint32_t outAddr, uint32_t inAddr, const char* Imms ,uint32_t size, uint32_t mode);

void NNPU_VctrDotProd(uint32_t outAddr, uint32_t in1Addr, uint32_t in2Addr, uint32_t size, 
                      uint32_t mode);

void NNPU_VctrReduceSum(uint32_t outAddr, uint32_t inAddr, uint32_t size, 
                      uint32_t mode);

void NNPU_VctrReduceMax(uint32_t outAddr, uint32_t inAddr, uint32_t size, 
                      uint32_t mode);

void NNPU_VctrReduceMin(uint32_t outAddr, uint32_t inAddr, uint32_t size, 
                      uint32_t mode);

void NNPU_MAddM(uint32_t outAddr, uint32_t outRowStride, 
                uint32_t in1Addr, uint32_t in1RowStride,
                uint32_t in2Addr, uint32_t in2RowStride,
                uint32_t nRow, uint32_t nCol, uint32_t mode);

void NNPU_MSubM(uint32_t outAddr, uint32_t outRowStride, 
                uint32_t in1Addr, uint32_t in1RowStride,
                uint32_t in2Addr, uint32_t in2RowStride,
                uint32_t nRow, uint32_t nCol, uint32_t mode);

void NNPU_MMulM(uint32_t outAddr, uint32_t outRowStride, 
                uint32_t in1Addr, uint32_t in1RowStride,
                uint32_t in2Addr, uint32_t in2RowStride,
                uint32_t nRow, uint32_t nCol, uint32_t mode);

void NNPU_MReduceSumRow(uint32_t outAddr, uint32_t inAddr, uint32_t inRowStride,
                        uint32_t nRow, uint32_t nCol, uint32_t mode);

void NNPU_MAddV(uint32_t outAddr, uint32_t outRowStride, 
                uint32_t matAddr, uint32_t matRowStride,
                uint32_t vctrAddr, uint32_t nRow, uint32_t nCol, uint32_t mode);

void NNPU_MSubV(uint32_t outAddr, uint32_t outRowStride,
                uint32_t matAddr, uint32_t matRowStride,
                uint32_t vctrAddr, uint32_t nRow, uint32_t nCol, uint32_t mode);

void NNPU_MMulV(uint32_t outAddr, uint32_t outRowStride,
                uint32_t matAddr, uint32_t matRowStride,
                uint32_t vctrAddr, uint32_t nRow, uint32_t nCol, uint32_t mode);

void NNPU_MRowDot(uint32_t outAddr, uint32_t in1Addr, uint32_t in1RowStride,
                  uint32_t in2Addr, uint32_t in2RowStride,
                  uint32_t nRow, uint32_t nCol, uint32_t mode); 

void NNPU_VAddS(uint32_t outAddr, uint32_t vctrAddr, uint32_t sclrAddr, 
                uint32_t size, uint32_t mode);

void NNPU_VSubS(uint32_t outAddr, uint32_t vctrAddr, uint32_t sclrAddr, 
                uint32_t size, uint32_t mode);

void NNPU_VMulS(uint32_t outAddr, uint32_t vctrAddr, uint32_t sclrAddr, 
                uint32_t size, uint32_t mode);

void NNPU_VDivS(uint32_t outAddr, uint32_t vctrAddr, uint32_t sclrAddr, 
                uint32_t size, uint32_t mode);

void NNPU_VGTMS(uint32_t outAddr, uint32_t vctrAddr, uint32_t sclrAddr, 
                uint32_t size, uint32_t mode);

void NNPU_SSubV(uint32_t outAddr, uint32_t vctrAddr, uint32_t sclrAddr, 
                uint32_t size, uint32_t mode);

void NNPU_SDivV(uint32_t outAddr, uint32_t vctrAddr, uint32_t sclrAddr, 
                uint32_t size, uint32_t mode);

void NNPU_ScratchpadCopy(uint32_t dstAddr, int32_t dstOffset, uint32_t dstStride,
                         uint32_t srcAddr, int32_t srcOffset, uint32_t dsrStride,
                         uint32_t elemBytes, uint32_t nElem);

void NNPU_Memset(uint32_t addr, uint32_t nUnit, uint32_t stride, const char *val, uint32_t mode);

void NNPUSynchronize(uint32_t timeout);

#ifdef __cplusplus
}
#endif

#endif