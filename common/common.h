#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <cuda_runtime.h>

#define CUDA_CHECK(cmd)                 \
do {                                    \
  cudaError_t status = (cmd);           \
  if (cudaSuccess != status) {          \
    std::cerr << "["                    \
      << __FILE__ << ":"                \
      << __LINE__ << "] "               \
      << cudaGetErrorName(status)       \
      << ", "                           \
      << cudaGetErrorString(status)     \
      << std::endl;                     \
    std::abort();                       \
  }                                     \
} while(0)


inline auto gpu_malloc(size_t size) {
  char *ptr = nullptr;
  cudaError_t status = cudaMalloc(&ptr, size);
  return std::make_pair(ptr, status);
}

struct gpu_deleter {
  void operator() (void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
};

