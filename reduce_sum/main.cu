#include <cstdlib>
#include <stdio.h>
//#include <cuda_profiler_api.h>

#include "common.h"
#include "reduce.h"

// NOTE: block size is a parameter to be fine tuned!
// block size is limited: 64, 128, 256, 512, 1024
// 256 is fine tuned parameter on (mx150, tesla v100) gpu
constexpr int kBlockSize = 256;

// NOTE:for different gpus, should set proper value of num_blocks, larger is better.
constexpr size_t num_blocks = (1ULL << 20); // 1G

// check results, template parameter N means the number of sum of block per thread block
template <int N>
void CheckResults(const char *kernel_name, int *h_out, int *d_out, int nRepeats, float totalTime) {
  CUDA_CHECK(cudaMemcpy(h_out, d_out, num_blocks*sizeof(int), cudaMemcpyDeviceToHost));

  for (size_t i=0; i<num_blocks/N; ++i) {
    if (h_out[i] != N * kBlockSize) {
      fprintf(stderr, "check error: %ld, %d, %d\n", i, h_out[i], N*kBlockSize);
      std::abort();
    }
  }

  float throughput = sizeof(int) * nRepeats * num_blocks * kBlockSize / 1024. / 1024. /1024. / totalTime * 1000;
  printf("%s: %.3f ms, %.3f GB/s\n", kernel_name, totalTime/nRepeats, throughput);
}

int main() {

  //cudaProfilerStart();
  // put the profiled code here!
  //cudaProfilerStop();

  // events for timing
  cudaEvent_t startEvent, stopEvent;
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));

  const size_t N = kBlockSize * num_blocks;
  const size_t nsize = N * sizeof(int);

  int *h_array, *h_out;
  h_array = (int *)malloc(nsize);
  if (h_array == NULL) 
    return 1;

  for (int i=0; i<N; ++i) {
    h_array[i] = 1;
  }

  h_out = (int *) malloc(nsize / kBlockSize);
  if (h_out == NULL) 
    return 2;

  int *d_array, *d_out, *d_sum;
  CUDA_CHECK(cudaMalloc(&d_array, nsize));
  CUDA_CHECK(cudaMalloc(&d_out, num_blocks * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_out, 0, kBlockSize));

  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_array, h_array, nsize, cudaMemcpyHostToDevice));

  int block_size = kBlockSize;
  int grid_size = (N + kBlockSize - 1)/kBlockSize;

  constexpr int kRepeats = 10;
  float totalTime = 0;
  
  // warm up
  reduce_v1<kBlockSize><<<grid_size, block_size>>>(d_array, d_out);
  reduce_v2<kBlockSize><<<grid_size, block_size>>>(d_array, d_out);
  reduce_v3<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_out);
  reduce_v4<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_out);
  reduce_v5<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_out);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v1<kBlockSize><<<grid_size, block_size>>>(d_array, d_out);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  CheckResults<1>("kernel v1", h_out, d_out, kRepeats, totalTime);

  totalTime = 0;
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v2<kBlockSize><<<grid_size, block_size>>>(d_array, d_out);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  CheckResults<1>("kernel v2", h_out, d_out, kRepeats, totalTime);

  totalTime = 0;
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v3<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_out);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  CheckResults<2>("kernel v3", h_out, d_out, kRepeats, totalTime);


  totalTime = 0;
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v4<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_out);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  CheckResults<2>("kernel v4", h_out, d_out, kRepeats, totalTime);


  totalTime = 0;
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v5<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_out);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  CheckResults<2>("kernel v5", h_out, d_out, kRepeats, totalTime);

  totalTime = 0;
  constexpr int kNfold = 32;
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v6<kBlockSize><<<grid_size/kNfold, block_size>>>(d_array, d_out, kNfold/2);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  CheckResults<kNfold>("kernel v6", h_out, d_out, kRepeats, totalTime);


  // free memory
  free(h_array);
  free(h_out);
  CUDA_CHECK(cudaFree(d_array));
  CUDA_CHECK(cudaFree(d_out));

  return 0;
}

