#include <iostream>

#include "gpuinfo.h"

int main() {

  std::cout << GetGpuInfo()->GetGpuNums() << std::endl;
  std::cout << GetGpuInfo()->GetComputeCapability() << std::endl;


  return 0;
}
