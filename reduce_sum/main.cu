#include <stdio.h>
#include <cuda_profiler_api.h>

#include "common.h"
#include "reduce.h"

int main() {

  // block size 1024
  constexpr int kBlockSize = 1024;

  // events for timing
  cudaEvent_t startEvent, stopEvent;
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));


  // NOTE:for different gpu, should set proper value of num_blocks.
  constexpr size_t num_blocks = 102400; 

  const size_t N = 1024*num_blocks;
  const size_t nsize = N * sizeof(int);

  int *h_array;
  h_array = (int *)malloc(nsize);
  for (int i=0; i<N; ++i) {
    h_array[i] = 1;
  }

  int *d_array, *d_r1, *d_sum;
  CUDA_CHECK(cudaMalloc(&d_array, nsize));
  CUDA_CHECK(cudaMalloc(&d_r1, kBlockSize * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_r1, 0, kBlockSize));


  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_array, h_array, nsize, cudaMemcpyHostToDevice));

  int block_size = kBlockSize;
  int grid_size = (N + kBlockSize - 1)/kBlockSize;

  constexpr int kRepeats = 10;
  
  // warm up
  reduce_v1<kBlockSize><<<grid_size, block_size>>>(d_array, d_r1);
  reduce_v2<kBlockSize><<<grid_size, block_size>>>(d_array, d_r1);
  reduce_v3<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_r1);
  reduce_v4<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_r1);
  reduce_v5<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_r1);

  float totalTime = 0;
  float throughput = 0;
  //cudaProfilerStart();
  //cudaProfilerStop();
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v1<kBlockSize><<<grid_size, block_size>>>(d_array, d_r1);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  throughput = kRepeats * nsize / 1024. / 1024. /1024. / totalTime * 1000;
  printf("kernel v1: %.3f ms, %.3f GB/s\n", totalTime/kRepeats, throughput);

  totalTime = 0;
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v2<kBlockSize><<<grid_size, block_size>>>(d_array, d_r1);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  throughput = kRepeats * nsize / 1024. / 1024. /1024. / totalTime * 1000;
  printf("kernel v2: %.3f ms, %.3f GB/s\n", totalTime/kRepeats, throughput);

  totalTime = 0;
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v3<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_r1);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  throughput = kRepeats * nsize / 1024. / 1024. /1024. / totalTime * 1000;
  printf("kernel v3: %.3f ms, %.3f GB/s\n", totalTime/kRepeats, throughput);


  totalTime = 0;
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v4<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_r1);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  throughput = kRepeats * nsize / 1024. / 1024. /1024. / totalTime * 1000;
  printf("kernel v4: %.3f ms, %.3f GB/s\n", totalTime/kRepeats, throughput);

  totalTime = 0;
  for (int i=0; i < kRepeats; ++i) {
    cudaEventRecord(startEvent, 0);
    reduce_v5<kBlockSize><<<grid_size/2, block_size>>>(d_array, d_r1);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    totalTime += ms;
  }
  throughput = kRepeats * nsize / 1024. / 1024. /1024. / totalTime * 1000;
  printf("kernel v5: %.3f ms, %.3f GB/s\n", totalTime/kRepeats, throughput);

  // free
  free(h_array);
  CUDA_CHECK(cudaFree(d_array));
  CUDA_CHECK(cudaFree(d_r1));

  return 0;
}

