#pragma once

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <cuda_runtime.h>

std::mutex g_mutex;

#define CUDA_CHECK(cmd)                     \
do {                                        \
  cudaError_t status = (cmd);               \
  if (cudaSuccess != status) {              \
    std::lock_guard<std::mutex> g(g_mutex); \
    std::cerr << "["                        \
      << __FILE__ << ":"                    \
      << __LINE__ << "] "                   \
      << cudaGetErrorName(status)           \
      << ", "                               \
      << cudaGetErrorString(status)         \
      << std::endl;                         \
    std::abort();                           \
  }                                         \
} while(0)

