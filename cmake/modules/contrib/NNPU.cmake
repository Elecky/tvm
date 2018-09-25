# CMake Build rules for NNPU
find_program(PYTHON NAMES python python3 python3.6)

if(PYTHON)
    set(NNPU_RUNTIME_SRCS nnpu/src/device_api.cpp nnpu/src/runtime.cpp nnpu/src/sim_driver.cpp)
    add_library(nnpu SHARED ${NNPU_RUNTIME_SRCS})
    target_include_directories(nnpu PUBLIC nnpu/include)
else()
  message(STATUS "Cannot found python in env, VTA build is skipped..")
endif()