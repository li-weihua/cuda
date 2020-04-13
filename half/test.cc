#include "fp16.h"

#include <iostream>
#include <chrono>
#include <unistd.h>

using namespace std;

// main function to measure elapsed time of a C++ program 
// using chrono library
int main() {

  const int n = 8 * 10000000;
  const size_t nsize = sizeof(int) * n;
  const size_t msize = sizeof(uint16_t) * n;

  float *x = (float *)aligned_alloc(32, nsize);

  uint16_t *y = (uint16_t *)aligned_alloc(32, msize); 
  uint16_t *z = (uint16_t *)aligned_alloc(32, msize); 

  for (int i=0; i<n; ++i) {
    x[i] = n % 65535;
  }


	auto start = chrono::steady_clock::now();

  for (int i=0; i<n; ++i) {
    y[i] = FastFloatToHalf(x[i]);
  }

	auto end = chrono::steady_clock::now();

  float time = chrono::duration_cast<chrono::milliseconds>(end - start).count();

	cout << "Elapsed time in milliseconds : " 
		<< time
		<< " ms" << endl;

	start = chrono::steady_clock::now();

  FloatToHalf_AVX(z, x, n);

	end = chrono::steady_clock::now();
  time = chrono::duration_cast<chrono::milliseconds>(end - start).count();

	cout << "Elapsed time in milliseconds : " 
		<< time
		<< " ms" << endl;

  // check diff
  int maxdiff = 0;

  for (int i=0; i<n; ++i) {
    int diff = abs(y[i] - z[i]);
    if (diff > maxdiff) {
      maxdiff = diff;
    }
  }

  std::cout << "maxdiff = " << maxdiff << std::endl;

  free(x);
  free(y);
  free(z);

	return 0;
}
