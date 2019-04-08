cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")
set(SIM_SRC ../../NNPUSim/src/)
# set(NNPU_INSN_SRC ../../../src/insn.cpp ../../../src/insn_ctors.cpp)

# include directories
include_directories("../../../include")
include_directories("../../../3rdparty/dlpack/include")
include_directories("../../../3rdparty/dmlc-core/include")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

add_executable(nnpu_test test_main.cpp)
target_include_directories(nnpu_test PUBLIC ../../include ../../NNPUSim/include)