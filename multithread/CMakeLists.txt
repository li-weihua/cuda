cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(test_cuda CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Common)
LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

message("generate architectures:\n ${ARCH_FLAGS}")

add_executable(test_cuda main.cc add.cu macro.h)

target_include_directories(test_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

