cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
set(SIM_SRC ../../NNPUSim/src/)
set(NNPU_S0SIM_SRC test_main.cpp ${SIM_SRC}/S0Simulator.cpp ${SIM_SRC}/bit_packer.cpp
                   ${SIM_SRC}/bit_packer_factory.cpp ${SIM_SRC}/bit_wrapper.cpp)
# include directories
include_directories("../../../include")
include_directories("../../../dlpack/include")
include_directories("../../../dmlc-core/include")
add_executable(nnpu_test ${NNPU_S0SIM_SRC})
target_include_directories(nnpu_test PUBLIC ../../include /usr/local/include ../../NNPUSim/include)

target_link_libraries(nnpu_test -lyaml-cpp)