#pragma region case: minmax-init-list
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 85

#include <algorithm>
#include <cstdio>
#include <initializer_list>

#include "test_utils.h"

int main() {
  const int result = __polyregion_offload_f1__([=]() {
    const int lo = std::min({7, 2, 5});
    const int hi = std::max(std::initializer_list<int>{3, 6, 4});
    const bool blo = std::min({true, false, true});
    const bool bhi = std::max(std::initializer_list<bool>{false, true, false});
    const float flo = std::min({-0.0f, 0.0f});
    const float fhi = std::max(std::initializer_list<float>{0.0f, -0.0f});
    int evaluation = 0;
    const int snapshot = std::min({++evaluation, ++evaluation, ++evaluation});
    int order = 0;
    const int ordered = std::max({(order = order * 10 + 1, 1), (order = order * 10 + 2, 2), (order = order * 10 + 3, 3)});
    return std::min(lo, hi) + std::max(lo, hi) + 1 + (!blo && bhi ? 40 : 0) + (1.0f / flo < 0.0f ? 10 : 0) + (1.0f / fhi > 0.0f ? 20 : 0)
           + (snapshot == 1 && evaluation == 3 && ordered == 3 && order == 123 ? 6 : 0);
  });
  std::printf("%d", result);
  return 0;
}
