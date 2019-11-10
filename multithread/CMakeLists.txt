cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(test_cuda CUDA CXX)

enable_language(CUDA)

add_executable(test_cuda main.cc add.cu)

target_include_directories(test_cuda PUBLIC 
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                          )
