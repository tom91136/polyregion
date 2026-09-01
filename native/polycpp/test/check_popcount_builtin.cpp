#pragma region case: builtin-popcount
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 16

#include <cstdio>

#include "test_utils.h"

static unsigned populationCount(const unsigned value) { return __builtin_popcount(value); }

int main() {
  const unsigned value = 0xf0f00f0fu;
  const unsigned result = __polyregion_offload_f1__([=]() { return populationCount(value); });
  std::printf("%u", result);
  return 0;
}
