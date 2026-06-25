#pragma region case: positional-and-designated
#pragma region using: num_type=double,float
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_NUM_TYPE={num_type} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 10

#include <cstdio>

#include "test_utils.h"

#ifndef CHECK_NUM_TYPE
  #define CHECK_NUM_TYPE double
#endif

struct Pt {
  CHECK_NUM_TYPE x;
  CHECK_NUM_TYPE y;
};

int main() {
  CHECK_NUM_TYPE r = __polyregion_offload_f1__([=]() {
    CHECK_NUM_TYPE a = Pt{3.0, 4.0}.x + Pt{3.0, 4.0}.y;                     // positional: 7
    CHECK_NUM_TYPE b = Pt{.x = 1.0, .y = 2.0}.x + Pt{.x = 1.0, .y = 2.0}.y; // designated: 3
    return a + b;                                                           // 7 + 3 = 10
  });
  printf("%.0f", r);
  return 0;
}
