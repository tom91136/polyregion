#pragma region case: pragma-unroll
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_ATTRIBUTE=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 15

#pragma region case: likely
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -std=c++20 -DCHECK_ATTRIBUTE=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

#include "test_utils.h"

int main() {
  int result = __polyregion_offload_f1__([=]() {
#if CHECK_ATTRIBUTE == 1
    int sum = 0;
  #pragma unroll
    for (int i = 0; i < 6; ++i)
      sum += i;
    return sum;
#elif CHECK_ATTRIBUTE == 2
    int value = 0;
    if (true) [[likely]] {
      value = 42;
    }
    return value;
#else
  #error "CHECK_ATTRIBUTE undefined"
#endif
  });
  std::printf("%d", result);
  return 0;
}
