#include <iostream>
#include <thread>
#include <mutex>

#include <cuda_runtime.h>

#include "macro.h"

void add(int*, int, int);

void threadFunc(int device, int a, int b, int &c) {

  CUDA_CHECK(cudaSetDevice(device));

  int *x;
  CUDA_CHECK(cudaMalloc(&x, sizeof(int)));

  add(x, a, b);

  CUDA_CHECK(cudaMemcpy(&c, x, sizeof(int), cudaMemcpyDeviceToHost));
}

int main() {

  int c1 = 0;
  int c2 = 0;

  std::thread t1(threadFunc, 0, 1, 3, std::ref(c1));
  std::thread t2(threadFunc, 1, 2, 7, std::ref(c2));

  t1.join();
  t2.join();

  std::cout << c1 << std::endl;
  std::cout << c2 << std::endl;

  return 0;
}

