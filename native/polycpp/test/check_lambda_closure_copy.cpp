#pragma region case: closure-copy
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 21

#include <cstdio>

#include "test_utils.h"

int main() {
  int base = 20;
  const int result = __polyregion_offload_f1__([=]() {
    auto add = [base](int x) { return base + x; };
    auto addCopy = add;
    return addCopy(1);
  });
  std::printf("%d", result);
  return 0;
}
