#pragma once

class GpuInfo {
public:
  int GetGpuNums() const;
  int GetComputeCapability() const;

private:
  int xx;
};

GpuInfo* GetGpuInfo();
