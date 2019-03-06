# CMake Build rules for NNPU
find_program(PYTHON NAMES python python3 python3.6)

if(PYTHON)
  set (CMAKE_PREFIX_PATH /opt/systemc)
  find_package(SystemCLanguage CONFIG REQUIRED)

  set(RUNTIME_SRC nnpu/src/)
  set(NNPU_RUNTIME_SRCS ${RUNTIME_SRC}/device_api.cpp ${RUNTIME_SRC}/runtime.cpp 
                        ${RUNTIME_SRC}/sim_driver.cpp ${RUNTIME_SRC}/insn.cpp
                        ${RUNTIME_SRC}/insn_ctors.cpp)

  set(SIM_SRC nnpu/NNPUSim/src/)
  set(NNPU_SIM_COMMON ${SIM_SRC}/common/bit_packer.cpp ${SIM_SRC}/common/bit_packer_factory.cpp
                      ${SIM_SRC}/common/bit_wrapper.cpp ${SIM_SRC}/common/wire.cpp
                      ${SIM_SRC}/common/data_types.cpp
                      )

  set(NNPU_S0SIM_SRC ${SIM_SRC}/ram.cpp ${SIM_SRC}/S0Simulator.cpp)
  set(NNPU_S1SIM_SRC ${SIM_SRC}/sim_module.cpp ${SIM_SRC}/insn_mem.cpp
                     ${SIM_SRC}/insn_wrapper.cpp ${SIM_SRC}/insn_decoder.cpp
                     ${SIM_SRC}/controller.cpp ${SIM_SRC}/reservation_station.cpp
                     ${SIM_SRC}/alu.cpp ${SIM_SRC}/branch_unit.cpp ${SIM_SRC}/reg_file_mod.cpp
                     ${SIM_SRC}/load_store_unit.cpp ${SIM_SRC}/sclr_buffer.cpp
                     ${SIM_SRC}/future_file.cpp ${SIM_SRC}/memory_queue.cpp
                     ${SIM_SRC}/data_read_unit.cpp ${SIM_SRC}/address_range.cpp
                     ${SIM_SRC}/vctr_calc_unit.cpp ${SIM_SRC}/data_write_unit.cpp
                     ${SIM_SRC}/mat_calc_unit.cpp ${SIM_SRC}/DMA_copy_buffer_LS_unit.cpp
                     ${SIM_SRC}/buffer_copy_set_unit.cpp ${SIM_SRC}/s1_simulator.cpp
                     ${SIM_SRC}/mat_write_back_unit.cpp)
  
  set(NNPU_SCSIM_DIR ${SIM_SRC}/sc_sim)
  set(NNPU_SCSIM_SRCS ${NNPU_SCSIM_DIR}/sc_simulator.cpp ${NNPU_SCSIM_DIR}/ifetch.cpp
                      ${NNPU_SCSIM_DIR}/idecode.cpp ${NNPU_SCSIM_DIR}/issue_queue.cpp
                      ${NNPU_SCSIM_DIR}/future_file.cpp ${NNPU_SCSIM_DIR}/reserve_station.cpp)

  add_library(nnpu-dummy SHARED ${NNPU_SCSIM_DIR}/dummy.cpp)

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