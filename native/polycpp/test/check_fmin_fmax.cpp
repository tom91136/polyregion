#pragma region case: fmin-fmax
#pragma region using: num_type=double,float
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_NUM_TYPE={num_type} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3.0 7.0 2.5

#include <cmath>
#include <cstdio>

#include "test_utils.h"

#ifndef CHECK_NUM_TYPE
  #define CHECK_NUM_TYPE double
#endif

int main() {
  using T = CHECK_NUM_TYPE;
  T a = __polyregion_offload_f1__([]() { return std::fmin(T(3), T(5)); });                    // 3
  T b = __polyregion_offload_f1__([]() { return std::fmax(T(7), T(2)); });                    // 7
  T c = __polyregion_offload_f1__([]() { return std::fmin(std::fmax(T(2.5), T(1)), T(4)); }); // 2.5
  printf("%.1f %.1f %.1f", a, b, c);
  return 0;
}
