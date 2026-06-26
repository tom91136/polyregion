#pragma region case: offload-local-array
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

#include "test_utils.h"

int main() {
  const int r = __polyregion_offload_f1__([]() -> int {
    [[clang::annotate("__polyregion_local")]] int scratch[256];
    scratch[__polyregion_builtin_gpu_local_idx(0)] = 42;
    return scratch[0];
  });
  std::printf("%d", r);
  return 0;
}
