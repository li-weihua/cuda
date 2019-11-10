#include <iostream>

#include <cuda_runtime.h>

void add(int*, int, int);

int main() {

  int a = 1;
  int b = 2;
  int c = 0;

  int *x;
  cudaMalloc(&x, sizeof(int));

  add(x, a, b);

  cudaMemcpy(&c, x, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << c << std::endl;

  return 0;
}

