cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")
set(SIM_SRC ../../NNPUSim/src/)
set(NNPU_INSN_SRC ../../../src/insn.cpp)
set(NNPU_S1SIM_SRC test_main.cpp ${SIM_SRC}/common/bit_packer.cpp
                   ${SIM_SRC}/common/bit_packer_factory.cpp ${SIM_SRC}/common/bit_wrapper.cpp
                   ${SIM_SRC}/common/wire.cpp ${SIM_SRC}/sim_module.cpp ${SIM_SRC}/insn_mem.cpp
                   ${SIM_SRC}/insn_wrapper.cpp ${SIM_SRC}/insn_decoder.cpp
                   ${SIM_SRC}/controller.cpp ${SIM_SRC}/reg_file_mod.cpp
                   ${SIM_SRC}/reservation_station.cpp ${SIM_SRC}/alu.cpp
                   ${SIM_SRC}/branch_unit.cpp ${SIM_SRC}/load_store_unit.cpp
                   ${SIM_SRC}/sclr_buffer.cpp ${SIM_SRC}/ram.cpp 
                   ${SIM_SRC}/future_file.cpp ${SIM_SRC}/memory_queue.cpp
                   ${SIM_SRC}/address_range.cpp ${SIM_SRC}/data_read_unit.cpp
                   ${SIM_SRC}/vctr_calc_unit.cpp ${SIM_SRC}/data_write_unit.cpp
                   ${SIM_SRC}/DMA_copy_buffer_LS_unit.cpp
                   ${SIM_SRC}/buffer_copy_set_unit.cpp)
# include directories
include_directories("../../../include")
include_directories("../../../dlpack/include")
include_directories("../../../dmlc-core/include")
add_executable(nnpu_test ${NNPU_S1SIM_SRC} ${NNPU_INSN_SRC})
target_include_directories(nnpu_test PUBLIC ../../include /usr/local/include ../../NNPUSim/include)

target_link_libraries(nnpu_test -lyaml-cpp)