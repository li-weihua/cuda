#include <cuda_runtime.h>

#include "gpuinfo.h"

int GpuInfo::GetGpuNums() const {
  int n = 0;
  cudaGetDeviceCount(&n);
  return n;
}

int GpuInfo::GetComputeCapability() const {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);

  return p.major * 10 + p.minor;
}

static GpuInfo g_GpuInfo;

GpuInfo* GetGpuInfo() {
  return &g_GpuInfo;
}
