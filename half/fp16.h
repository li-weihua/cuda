/*
 * single-precision floating number is represented as
 *      +---+---------+----------------------------+
 *      | S |EEEE EEEE|MMM MMMM MMMM MMMM MMMM MMMM|
 *      +---+---------+----------------------------+
 * Bits  31  23-30     0-22
 *
 *
 * half-precision floating number is represented as
 *      +---+------+------------+
 *      | S |E EEEE|MM MMMM MMMM|
 *      +---+------+------------+
 * Bits  15  10-14     0-9
 *
 * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0 - zero bits.
 * 
 * For 32-bit float, its bias is 2^7-1 = 127
 * For 16-bit float, its bias is 2^5-1 = 15 
 */
#pragma once

#include <cstdint>
#include <immintrin.h>

// typedef uint16_t half
// half is represented by unsigend short

// scalar version
inline uint16_t FastFloatToHalf(float f) {

  int32_t n = *reinterpret_cast<int32_t *>(&f);

  int32_t s = (n >> 31) << 15;

  int32_t m = (n >> 13) & 0x3FF; 

  int32_t e = ((n>>23) & 0xFF) - 127 + 15;

  if (e < 0) {
    e = 0;
    m = 0;
  }
  else if (e > 30) {
    e = 30;
  }

  return s | (e << 10) | m;
}

// simd version
//__m128i _mm256_cvtps_ph (__m256 a, _MM_FROUND_NO_EXC) 

// x and y are aligned!
void FloatToHalf_AVX(uint16_t *y, float *x, int n) {
  int m = n / 8;

  // vector version
  for (int i=0; i < m; ++i) {
    __m256 a = _mm256_load_ps(x);
    __m128i b = _mm256_cvtps_ph(a, _MM_FROUND_NO_EXC);
    _mm_store_si128((__m128i *)y, b);
    x += 8;
    y += 8;
  }

  int nleft = n - m*8;

  for (int i=0; i < nleft; ++i) {
    *y = FastFloatToHalf(*x);
    x += 1;
    y += 1;
  }
}


