#pragma region case: offload-spin-wait
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 8 8

#include <cstdint>
#include <cstdio>

#include "test_utils.h"

int main() {
  constexpr uint32_t lanes = 8;
  uint32_t flags[lanes] = {};
  uint32_t seen[lanes] = {};
  __polyregion_offload_workgroup__(lanes, [=, &flags, &seen](uint32_t lid) {
    __polyregion_gpu_volatile_store_u32(&flags[lid], lid + 1u);
    while (__polyregion_gpu_volatile_load_u32(&flags[lid]) == 0u) {
    }
    seen[lid] = __polyregion_gpu_volatile_load_u32(&flags[lid]);
  });
  int correct = 0;
  for (uint32_t i = 0; i < lanes; ++i)
    correct += seen[i] == i + 1u;
  std::printf("%d %d", correct, static_cast<int>(lanes));
  return 0;
}
