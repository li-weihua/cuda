cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(test_cuda CXX)

find_package(CUDA)

add_executable(test_pcie main.cc)

target_include_directories(test_pcie PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_pcie PUBLIC ${CUDA_LIBRARIES})

