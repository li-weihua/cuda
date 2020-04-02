cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(Tensorcore CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}   \
        -O3 -std=c++11 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        "
)

message("nvcc flags:\n ${CMAKE_CUDA_FLAGS}")

add_executable(tensorcore1 matmultiply_16x16.cu common.h)

