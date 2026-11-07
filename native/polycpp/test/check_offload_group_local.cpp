#pragma region case: offload-group-local
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 64 64

#include <cstdint>
#include <cstdio>

#include "test_utils.h"

int main() {
  constexpr uint32_t lanes = 64;
  int doubled[lanes], shifted[lanes];
  __polyregion_offload_workgroup__(lanes, [=, &doubled, &shifted](uint32_t global) {
    [[clang::annotate("__polyregion_local")]] int scratch[lanes];
    const uint32_t lid = __polyregion_gpu_local_idx(0);
    scratch[lid] = static_cast<int>(lid) * 2;
    doubled[global] = scratch[lid];
    shifted[lid] = static_cast<int>(__polyregion_gpu_global_idx(0)) + static_cast<int>(__polyregion_gpu_group_idx(0)) + 1;
  });
  int correctDoubled = 0, correctShifted = 0;
  for (uint32_t i = 0; i < lanes; ++i) {
    correctDoubled += doubled[i] == static_cast<int>(i) * 2;
    correctShifted += shifted[i] == static_cast<int>(i) + 1;
  }
  std::printf("%d %d", correctDoubled, correctShifted);
  return 0;
}
