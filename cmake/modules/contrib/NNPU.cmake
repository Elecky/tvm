# CMake Build rules for NNPU
find_program(PYTHON NAMES python python3 python3.6)

if(PYTHON)
  set (CMAKE_PREFIX_PATH /opt/systemc)
  find_package(SystemCLanguage CONFIG REQUIRED)

  set(RUNTIME_SRC nnpu/src/)
  set(NNPU_RUNTIME_SRCS ${RUNTIME_SRC}/device_api.cpp ${RUNTIME_SRC}/runtime.cpp 
                        ${RUNTIME_SRC}/sim_driver.cpp)

  set(SIM_SRC nnpu/NNPUSim/src/)
  set(NNPU_SIM_COMMON ${SIM_SRC}/common/bit_packer.cpp    ${SIM_SRC}/common/bit_packer_factory.cpp
                      ${SIM_SRC}/common/bit_wrapper.cpp   ${SIM_SRC}/common/data_types.cpp
                      ${SIM_SRC}/insn.cpp                 ${SIM_SRC}/misc/semaphore.cpp
                      ${SIM_SRC}/insn_ctors.cpp           ${SIM_SRC}/insn_functors.cpp)

  set(NNPU_S0SIM_SRC ${SIM_SRC}/ram.cpp ${SIM_SRC}/S0Simulator.cpp)
  set(NNPU_S1SIM_SRC ${SIM_SRC}/insn_wrapper.cpp 
                     ${SIM_SRC}/address_range.cpp ${SIM_SRC}/insn_opcode.cpp
                     ${SIM_SRC}/scratchpad_holder.cpp)
  
  set(NNPU_SCSIM_DIR ${SIM_SRC}/sc_sim)
  set(NNPU_SCSIM_EXEC_DIR ${NNPU_SCSIM_DIR}/instructions/)
  set(NNPU_SCSIM_EXEC_SRCS ${NNPU_SCSIM_EXEC_DIR}/gemm.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/mat_binary.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/mat_imm.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/mat_reduce_row.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/mat_row_dot.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/mat_vctr.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/vctr_binary.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/vctr_unary.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/vctr_imm.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/vctr_reduce.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/vctr_sclr.cpp
                           ${NNPU_SCSIM_EXEC_DIR}/vctr_dot_prod.cpp
                           )
  set(NNPU_SCSIM_SRCS ${NNPU_SCSIM_DIR}/sc_simulator.cpp            ${NNPU_SCSIM_DIR}/ifetch.cpp
                      ${NNPU_SCSIM_DIR}/idecode.cpp                 ${NNPU_SCSIM_DIR}/dispatch_queue.cpp
                      ${NNPU_SCSIM_DIR}/future_file.cpp             ${NNPU_SCSIM_DIR}/reserve_station.cpp
                      ${NNPU_SCSIM_DIR}/alu.cpp                     ${NNPU_SCSIM_DIR}/common_data_bus.cpp
                      ${NNPU_SCSIM_DIR}/branch_unit.cpp             ${NNPU_SCSIM_DIR}/load_store_unit.cpp
                      ${NNPU_SCSIM_DIR}/scalar_memory.cpp           
                      ${NNPU_SCSIM_DIR}/memory_queue.cpp            ${NNPU_SCSIM_DIR}/data_read_unit.cpp
                      ${NNPU_SCSIM_DIR}/data_write_unit.cpp         ${NNPU_SCSIM_DIR}/vctr_calc_unit.cpp
                      ${NNPU_SCSIM_DIR}/vector_unit.cpp             ${NNPU_SCSIM_DIR}/retire_bus.cpp
                      ${NNPU_SCSIM_DIR}/dram_access_unit.cpp        ${NNPU_SCSIM_DIR}/mat_calc_unit.cpp
                      ${NNPU_SCSIM_DIR}/matrix_unit.cpp             ${NNPU_SCSIM_DIR}/ram_access.cpp
                      ${NNPU_SCSIM_DIR}/ram_channel.cpp             ${NNPU_SCSIM_DIR}/buffer_access_spliter.cpp
                      ${NNPU_SCSIM_DIR}/mem_copy_set_unit.cpp       ${NNPU_SCSIM_DIR}/depend_queue.cpp
                      ${NNPU_SCSIM_DIR}/depend_queue_hub.cpp        ${NNPU_SCSIM_DIR}/pipeline_controller.cpp
                      ${NNPU_SCSIM_DIR}/pipeline_controller_impl.cpp
                      ${NNPU_SCSIM_EXEC_SRCS})

  add_library(nnpu-dummy SHARED ${NNPU_SCSIM_DIR}/dummy.cpp)
  set_target_properties(nnpu-dummy PROPERTIES CXX_VISIBILITY_PRESET default)

  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if (COMPILER_SUPPORTS_MARCH_NATIVE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
  
  add_library(nnpu SHARED ${NNPU_RUNTIME_SRCS} ${NNPU_SIM_COMMON} ${NNPU_S0SIM_SRC}
                          ${NNPU_S1SIM_SRC} ${NNPU_SCSIM_SRCS})
  target_include_directories(nnpu PUBLIC nnpu/include /usr/local/include nnpu/NNPUSim/include
                              /opt/systemc/include)
  target_link_libraries(nnpu -lyaml-cpp)
  target_link_libraries(nnpu SystemC::systemc)
else()
  message(STATUS "Cannot found python in env, NNPU build is skipped..")
endif()

file(GLOB RUNTIME_NNPU_SRCS src/runtime/nnpu/*.cc)
list(APPEND RUNTIME_SRCS ${RUNTIME_NNPU_SRCS})