#pragma region case: pointer-value-init
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#include <cstdio>

#include "test_utils.h"

int main() {
  const int r = __polyregion_offload_f1__([=]() {
    using Pointer = int *;
    Pointer pointer = Pointer();
    return pointer == nullptr ? 1 : 0;
  });
  std::printf("%d", r);
  return 0;
}
