cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")
set(SIM_SRC ../../NNPUSim/src/)
set(NNPU_INSN_SRC ../../src/insn.cpp)
set(NNPU_SIMDR_SRC ../../src/sim_driver.cpp)
set(NNPU_S0SIM_SRC  test_main.cpp 
                    ${SIM_SRC}/S0Simulator.cpp
                    ${SIM_SRC}/common/bit_packer.cpp
                    ${SIM_SRC}/common/bit_packer_factory.cpp
                    ${SIM_SRC}/common/bit_wrapper.cpp
                    ${SIM_SRC}/ram.cpp
                    ../../../src/runtime/registry.cc
                    ../../../src/runtime/c_runtime_api.cc
                    ../../../src/runtime/module.cc
                    ../../../src/runtime/file_util.cc
                    )
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin
    CACHE PATH "Single directory for all executable.")
# include directories
include_directories("../../../include")
include_directories("../../../dmlc-core/include")
include_directories("../../../src/runtime")
include_directories("../../../3rdparty/dlpack/include")
add_executable(nnpu_test ${NNPU_S0SIM_SRC} ${NNPU_INSN_SRC} ${NNPU_SIMDR_SRC})
target_include_directories(nnpu_test PUBLIC ../../include /usr/local/include ../../NNPUSim/include)

target_link_libraries(nnpu_test -lyaml-cpp)