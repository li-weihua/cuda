#include <stdio.h>
#include <cuda_runtime.h>

#include "common.h"

// kernel
__global__ void texture1d_kernel(float* output, 
                                 cudaTextureObject_t texObj, 
                                 float shift, 
                                 int n)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n) {
    // fetch original value
    output[i] = tex1D<float>(texObj, i + shift);
  }
}

// Host code
int main()
{
  // input
  const int n = 8;
  const int nsize = sizeof(float) * n;
  float *h_input = new float[n];
  for (int i=0; i<n; ++i) {
    h_input[i] = i+1;
  }

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray *cuArray;
  CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, n, 1));
  //CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, h_input, nsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_input, nsize, nsize, 1, cudaMemcpyHostToDevice));

  // output
  const int m = n;
  const int msize = sizeof(float) * m;
  float *h_output = new float[m];

  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_output, msize));


  // texture resource
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // texture description
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;

  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

  int blockSize = 16;
  int gridSize = (n + blockSize - 1) / blockSize;

  float shift = 0.5f; // fetch original value

  texture1d_kernel<<<gridSize, blockSize>>>(d_output, texObj, shift, n);

  CUDA_CHECK(cudaMemcpy(h_output, d_output, msize, cudaMemcpyDeviceToHost));

  for (int i=0; i<m; ++i) {
    printf("%.1f ", h_output[i]);
  }
  printf("\n");

  // free resource
  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);
  cudaFree(d_output);

  delete[] h_input;
  delete[] h_output;

  return 0;
}
