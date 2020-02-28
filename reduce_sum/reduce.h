#pragma once
// cuda code for block reduction
// suported block size is 64, 128, 256, 512, 1024
 
// version 1
// use shared memory
template <int BlockSize>
__global__ void reduce_v1(int *arr, int *odata) {

  __shared__ int s[BlockSize];

  int tid = threadIdx.x;
  int t = threadIdx.x + BlockSize * blockIdx.x;

  s[tid] = arr[t];
  __syncthreads();

  for (int i=BlockSize/2; i > 0; i >>= 1) {
    if (tid < i) {
      s[tid] += s[tid + i];
    }
     __syncthreads();
  }

  if (tid == 0) {
    odata[blockIdx.x] = s[0];
  }
}


// version 2
template <int BlockSize>
__global__ void reduce_v2(int *arr, int *odata) {

  __shared__ int s[BlockSize];

  int tid = threadIdx.x;
  int t = threadIdx.x + BlockSize * blockIdx.x;

  s[tid] = arr[t];
  __syncthreads();

  for (int i=BlockSize/2; i >= 32; i >>= 1) {
    if (tid < i) {
      s[tid] += s[tid + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    int sum = 0;
    for (int i=0; i<32; ++i) {
      sum += s[i];
    }

    odata[blockIdx.x] = sum;
  }
}

// 
template <int BlockSize>
__global__ void reduce_v3(int *arr, int *odata) {

  __shared__ int s[BlockSize];

  int tid = threadIdx.x;
  int t = threadIdx.x + BlockSize * 2 * blockIdx.x;

  s[tid] = arr[t] + arr[t + BlockSize];
  __syncthreads();

  for (int i=BlockSize/2; i >= 32; i >>= 1) {
    if (tid < i) {
      s[tid] += s[tid + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    int sum = 0;
    for (int i=0; i<32; ++i) {
      sum += s[i];
    }
    odata[blockIdx.x] = sum;
  }
}

// warp reduce
__device__ void warpReduce(volatile int* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

//
template <int BlockSize>
__global__ void reduce_v4(int *arr, int *odata) {

  __shared__ int s[BlockSize];

  int tid = threadIdx.x;
  int t = threadIdx.x + BlockSize * 2 * blockIdx.x;

  s[tid] = arr[t] + arr[t + BlockSize];
  __syncthreads();

  for (int i=BlockSize/2; i >= 64; i >>= 1) {
    if (tid < i) {
      s[tid] += s[tid + i];
    //__syncthreads();
    }
    __syncthreads();
  }
  /*
  if (tid < 512) { s[tid] += s[tid + 512]; } __syncthreads();
  if (tid < 256) { s[tid] += s[tid + 256]; } __syncthreads();
  if (tid < 128) { s[tid] += s[tid + 128]; } __syncthreads();
  if (tid < 64)  { s[tid] += s[tid + 64];  } __syncthreads();
  */

  // warp reduce
  if (tid < 32) {
    warpReduce(s, tid);
  }

  if (tid == 0) {
    odata[blockIdx.x] = s[0];
  }
}

//
template <int BlockSize>
__global__ void reduce_v5(int *arr, int *odata) {

  __shared__ int s[BlockSize];

  int tid = threadIdx.x;
  int t = threadIdx.x + BlockSize * 2 * blockIdx.x;

  s[tid] = arr[t] + arr[t + BlockSize];
  __syncthreads();

  for (int i=BlockSize/2; i >= 64; i >>= 1) {
    if (tid < i) {
      s[tid] += s[tid + i];
    }
    __syncthreads();
  }

  // warp shuffle
  if (tid < 32) {
    s[tid] += s[tid + 32];
    int v = s[tid];
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);

    if (tid == 0) {
      odata[blockIdx.x] = v;
    }
  }
}


