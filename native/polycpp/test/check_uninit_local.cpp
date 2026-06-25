#pragma region case: branch-assigned
#pragma region using: num_type=double,float
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_NUM_TYPE={num_type} -o {output} {input}
#pragma region do: A=1 {output}
#pragma region requires: 7
#pragma region do: A=-1 {output}
#pragma region requires: 99

#include <cstdio>
#include <cstdlib>

#include "test_utils.h"

#ifndef CHECK_NUM_TYPE
  #define CHECK_NUM_TYPE double
#endif

int main() {
  int a = std::atoi(std::getenv("A"));
  CHECK_NUM_TYPE result = __polyregion_offload_f1__([=]() {
    CHECK_NUM_TYPE x;
    if (a > 0) {
      x = 7.0;
    } else {
      x = 99.0;
    }
    return x;
  });
  printf("%.0f", result);
  return 0;
}
