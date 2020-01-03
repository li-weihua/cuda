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


auto gpu_malloc(size_t size) {
  char *ptr = nullptr;
  cudaError_t status = cudaMalloc(&ptr, size);
  return std::make_pair(ptr, status);
}

struct gpu_deleter {
  void operator() (void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
};

// @brief free host page-locked memory
struct host_deleter {
  void operator() (void *ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
};



int main() {

  constexpr size_t size = 1UL * 1024 * 1024 * 1024 ; // 1G

  char *h_ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_ptr, size, 0));

  std::unique_ptr<char, host_deleter> h_memory(h_ptr);

  auto gpu_memory = gpu_malloc(size);
  CUDA_CHECK(gpu_memory.second);

  std::unique_ptr<char, gpu_deleter> d_memory(gpu_memory.first);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int nloop = 10;
  for (int i=0; i<nloop; ++i) {
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(d_memory.get(), h_memory.get(), size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << (1*1000 / elapsedTime) << " GB/s" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
