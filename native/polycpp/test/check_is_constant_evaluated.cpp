#pragma region case: is_constant_evaluated
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

#include "test_utils.h"

int main() {
  int r = __polyregion_offload_f1__([]() { return __builtin_is_constant_evaluated() ? 999 : 42; });
  printf("%d", r);
  return 0;
}
