# CMake Build rules for NNPU
find_program(PYTHON NAMES python python3 python3.6)

if(PYTHON)
    set(RUNTIME_SRC nnpu/src/)
    set(NNPU_RUNTIME_SRCS ${RUNTIME_SRC}/device_api.cpp ${RUNTIME_SRC}/runtime.cpp 
                          ${RUNTIME_SRC}/sim_driver.cpp ${RUNTIME_SRC}/insn.cpp)
    set(SIM_SRC nnpu/NNPUSim/src/)
    set(NNPU_S0SIM_SRC ${SIM_SRC}/S0Simulator.cpp)
    add_library(nnpu SHARED ${NNPU_RUNTIME_SRCS} ${NNPU_S0SIM_SRC})
    target_include_directories(nnpu PUBLIC nnpu/include /usr/local/include nnpu/NNPUSim/include)
    target_link_libraries(nnpu -lyaml-cpp)
else()
  message(STATUS "Cannot found python in env, NNPU build is skipped..")
endif()