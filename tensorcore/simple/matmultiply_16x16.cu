#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "common.h"

using namespace nvcuda;

__global__ void warpGemm(half *A, half *B, float *C) {

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  wmma::load_matrix_sync(a_frag, A, 16);
  wmma::load_matrix_sync(b_frag, B, 16);
  wmma::fill_fragment(c_frag, 0.0f);

  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

  wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

__global__ void float2half_mat(float *a, float *b, int n, half *ha, half *hb) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n) {
    ha[i] = __float2half(a[i]);
    hb[i] = __float2half(b[i]);
  }
}


int main (int argc, char **argv) {
  int dev = 0;

  if (argc == 1) {
    dev = 0;
  } else if (argc == 2) {
    dev = atoi(argv[1]);
  } else {
    std::cerr << "input parameter error!" << std::endl;
    std::cerr << "Usage: " << argv[0] << " [dev]" << std::endl;
    std::abort();
  }

  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

  if (deviceProp.major < 7) {
    std::cerr << "Tensor Core requires SM 7.0 or higher" << std::endl;
    std::abort();
  }

  const int N = 16;
  const int N2 = N * N;
  const size_t matSize = sizeof(float) * N * N;

  float *h_A = (float *) malloc(matSize);
  float *h_B = (float *) malloc(matSize);
  float *h_C = (float *) malloc(matSize);

  for (int i=0; i < N2; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 1.0f;
    h_C[i] = 0;
  }

  float *d_A = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, matSize));

  float *d_B = nullptr;
  CUDA_CHECK(cudaMalloc(&d_B, matSize));

  half *A = nullptr;
  CUDA_CHECK(cudaMalloc(&A, matSize/2));

  half *B = nullptr;
  CUDA_CHECK(cudaMalloc(&B, matSize/2));

  float *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_C, matSize));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, matSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, matSize, cudaMemcpyHostToDevice));

  float2half_mat<<<1, 16*16>>>(d_A, d_B, N2, A, B);

  warpGemm<<<1,32>>>(A, B, d_C);

  CUDA_CHECK(cudaGetLastError());
  
  CUDA_CHECK(cudaMemcpy(h_C, d_C, matSize, cudaMemcpyDeviceToHost));

  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j) {
      printf("%.0f ", h_C[i*N + j]);
    }
    printf("\n");
  }

  // free resource
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(A);
  cudaFree(B);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
