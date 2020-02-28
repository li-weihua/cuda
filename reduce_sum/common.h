#pragma once

#include <cstdlib>
#include <iostream>
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

