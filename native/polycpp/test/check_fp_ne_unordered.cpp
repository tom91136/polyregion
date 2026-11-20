#pragma region case: fp-ne-unordered
#pragma region using: num_type=double,float
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_NUM_TYPE={num_type} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 1 0

#include <cstdio>
#include <limits>

#include "test_utils.h"

#ifndef CHECK_NUM_TYPE
  #define CHECK_NUM_TYPE double
#endif

int main() {
  using T = CHECK_NUM_TYPE;
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const int ne_self = __polyregion_offload_f1__([=]() { return nan != nan ? 1 : 0; });
  const int ne_one = __polyregion_offload_f1__([=]() { return nan != T(1) ? 1 : 0; });
  const int eq_self = __polyregion_offload_f1__([=]() { return nan == nan ? 1 : 0; });
  std::printf("%d %d %d", ne_self, ne_one, eq_self);
  return 0;
}
