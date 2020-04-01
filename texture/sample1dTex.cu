#include <stdio.h>
#include <cuda_runtime.h>

#include <string>
#include <list>

#include "common.h"

// kernel
__global__ void texture1d_kernel(float* output, 
                                 cudaTextureObject_t texObj, 
                                 float shift, 
                                 int n,
                                 int padding)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n + 2*padding) 
  {
    // fetch original value
    output[i] = tex1D<float>(texObj, (float)(i + shift - padding)/n);
  }
}


const std::list<std::pair<cudaTextureAddressMode, std::string>> kAddressMode = 
   {{cudaAddressModeWrap, "wrap"},
    {cudaAddressModeClamp, "clamp"}, 
    {cudaAddressModeMirror, "mirror"},  // mirror mode only supported for normalized coordinates
    {cudaAddressModeBorder, "border"}}; // border(period) mode only supported for normalized coordinates

void printResult(const std::string &str, float *out, int n, int padding);

// Host code
int main()
{
  // input
  const int n = 4;
  const int nsize = sizeof(float) * n;
  float *h_input = new float[n];
  for (int i=0; i<n; ++i) 
  {
    h_input[i] = i+1;
  }

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray *cuArray;
  CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, n, 1));
  //CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, h_input, nsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_input, nsize, nsize, 1, cudaMemcpyHostToDevice));

  // output
  int padding = 4;
  const int m = n + 2*padding;
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
  //texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t texObj = 0;

  int blockSize = 16;
  int gridSize = (n + blockSize - 1) / blockSize;

  float shift = 0.5f; // +0.5f original value

  for (const auto &mode: kAddressMode) {
    texDesc.addressMode[0] = mode.first;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    texture1d_kernel<<<gridSize, blockSize>>>(d_output, texObj, shift, n, padding);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, msize, cudaMemcpyDeviceToHost));

    printResult(mode.second, h_output, m, padding);
  }

  // free resource
  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);
  cudaFree(d_output);

  delete[] h_input;
  delete[] h_output;

  return 0;
}


void printResult(const std::string &str, float *h_output, int m, int padding) {

  printf("%6s mode: ", str.c_str());

  for (int i=0; i<m; ++i) 
  {
    printf("%.1f", h_output[i]);

    if (i == padding-1) {
      printf(" | ");
    }
    else if (i == m - padding -1) {
      printf(" | ");
    } else {
      printf(" ");
    }
  }
  printf("\n");
}

