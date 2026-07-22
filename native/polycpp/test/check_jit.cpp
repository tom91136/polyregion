#pragma region case: specialise
#pragma region using: jit=static,dynamic
#pragma region using: type=int,float
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_TYPE={type} -fstdpar-jit={jit} -o {output} {input}
#pragma region do: POLYRT_JIT_CACHE=off POLYRT_JIT_SPECIALISE=1 POLYRT_JIT_SPECIALISE_HOT=1 {output}
#pragma region requires: 42.0 7.0

#include <cstdio>

#include "test_utils.h"

#ifndef CHECK_TYPE
  #define CHECK_TYPE int
#endif

static CHECK_TYPE invoke(const CHECK_TYPE value) {
  return __polyregion_offload_f1__([=]() { return value + CHECK_TYPE{1}; });
}

int main() {
  const auto first = invoke(CHECK_TYPE{41});
  const auto second = invoke(CHECK_TYPE{6});
  std::printf("%.1f %.1f", static_cast<double>(first), static_cast<double>(second));
  return 0;
}
