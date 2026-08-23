#pragma region case: call-prism-core
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7

#include <cstdio>
#include <cstdlib>

#include "test_utils.h"

int main(int argc, char **) {
  const int value = argc + 6;
  const int result = __polyregion_offload_f1__([=]() {
    if (value < 0) std::abort();
    if (value < 0) __builtin_trap();
    return value + __builtin_constant_p(value);
  });
  std::printf("%d", result);
  return 0;
}
