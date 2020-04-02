cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(Texture CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}   \
        -O3 -std=c++11 \
        -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        "
)

message("nvcc flags:\n ${CMAKE_CUDA_FLAGS}")

add_executable(address_mode sample1dTex.cu common.h)

