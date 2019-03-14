cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")
set(SIM_SRC ../../NNPUSim/src/)
set(NNPU_INSN_SRC ../../../src/insn.cpp ../../../src/insn_ctors.cpp)
set(NNPU_S1SIM_SRC test_main.cpp ${SIM_SRC}/common/bit_packer.cpp
                   ${SIM_SRC}/common/bit_packer_factory.cpp ${SIM_SRC}/common/bit_wrapper.cpp
                   ${SIM_SRC}/common/data_types.cpp ${SIM_SRC}/insn_wrapper.cpp
                   ${SIM_SRC}/sc_sim/idecode.cpp ${SIM_SRC}/sc_sim/ifetch.cpp
                   ${SIM_SRC}/sc_sim/issue_queue.cpp ${SIM_SRC}/sc_sim/future_file.cpp
                   ${SIM_SRC}/sc_sim/reserve_station.cpp ${SIM_SRC}/sc_sim/alu.cpp
                   ${SIM_SRC}/sc_sim/common_data_bus.cpp ${SIM_SRC}/sc_sim/branch_unit.cpp
                   ${SIM_SRC}/sc_sim/load_store_unit.cpp ${SIM_SRC}/sc_sim/scalar_memory.cpp
                   ${SIM_SRC}/sc_sim/address_generate_unit.cpp
                   ${SIM_SRC}/sc_sim/memory_queue.cpp
                   )
# include directories
include_directories("../../../include")
include_directories("../../../3rdparty/dlpack/include")
include_directories("../../../3rdparty/dmlc-core/include")
add_executable(nnpu_test ${NNPU_S1SIM_SRC} ${NNPU_INSN_SRC})
target_include_directories(nnpu_test PUBLIC ../../include /usr/local/include 
    ../../NNPUSim/include /opt/systemc/include)

target_link_libraries(nnpu_test -lyaml-cpp)
find_package(SystemCLanguage CONFIG REQUIRED)
target_link_libraries(nnpu_test SystemC::systemc)