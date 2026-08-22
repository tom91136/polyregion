#pragma region case: gpu-memory-intrinsics
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -Werror -Wno-error=unused-command-line-argument -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdint>
#include <cstdio>

#include "test_utils.h"

int main() {
  uint32_t xchg = 7u;
  uint32_t add = 7u;
  uint32_t sub = 7u;
  uint32_t min = 7u;
  uint32_t max = 7u;
  uint32_t bit_and = 7u;
  uint32_t bit_or = 7u;
  uint32_t bit_xor = 7u;
  uint32_t scalar = 11u;
  uint32_t result = 0u;
  __polyregion_offload_workgroup__(1, [&xchg, &add, &sub, &min, &max, &bit_and, &bit_or, &bit_xor, &scalar, &result](uint32_t) {
    if (__polyregion_gpu_global_idx(0) == 0) {
      auto sum = __polyregion_gpu_volatile_load_u32(&scalar);
      __polyregion_gpu_volatile_store_u32(&scalar, sum + 1u);
      sum += __polyregion_gpu_atomic_xchg_u32(&xchg, 9u);
      sum += __polyregion_gpu_atomic_add_u32(&add, 2u);
      sum += __polyregion_gpu_atomic_sub_u32(&sub, 1u);
      sum += __polyregion_gpu_atomic_min_u32(&min, 3u);
      sum += __polyregion_gpu_atomic_max_u32(&max, 19u);
      sum += __polyregion_gpu_atomic_and_u32(&bit_and, 0x1Fu);
      sum += __polyregion_gpu_atomic_or_u32(&bit_or, 0x20u);
      sum += __polyregion_gpu_atomic_xor_u32(&bit_xor, 0x03u);
      if (scalar == 12u && sum == 67u) result = 42u;
    }
  });
  const bool valid =
      result == 42u && xchg == 9u && add == 9u && sub == 6u && min == 3u && max == 19u && bit_and == 7u && bit_or == 39u && bit_xor == 4u;
  std::printf("%d", valid ? 42 : 0);
  return 0;
}
