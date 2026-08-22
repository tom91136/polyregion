#pragma region case: gpu-subgroup-intrinsics
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -Werror -Wno-error=unused-command-line-argument -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdint>
#include <cstdio>

#include "test_utils.h"

int main() {
  uint32_t result[32]{};
  uint32_t runtimeMask = UINT32_MAX;
  __polyregion_offload_workgroup__(32, [&result, &runtimeMask](uint32_t globalLane) {
    constexpr uint32_t clamp = 31;
    const auto lane = __polyregion_gpu_lane_idx();
    const auto size = __polyregion_gpu_subgroup_size();
    const auto active = __polyregion_gpu_ballot(runtimeMask, true);
    const auto down = __polyregion_gpu_shuffle_down_u32(lane, 1u, clamp, runtimeMask);
    const auto up = __polyregion_gpu_shuffle_up_u32(lane, 1u, clamp, runtimeMask);
    const auto index = __polyregion_gpu_shuffle_idx_u32(lane, 0u, clamp);
    const auto xored = __polyregion_gpu_shuffle_xor_u32(lane, 1u, clamp, runtimeMask);
    const auto segmented = __polyregion_gpu_shuffle_idx_u32(lane, 0u, 7u, runtimeMask);
    const auto masked = __polyregion_gpu_shuffle_idx_u32(lane, 0u, clamp, 0xFFFFu);
    const auto ballot = __polyregion_gpu_ballot(runtimeMask, lane == 0u);
    const auto any = __polyregion_gpu_vote_any(runtimeMask, lane == 0u);
    const auto all = __polyregion_gpu_vote_all(runtimeMask, lane < size);
    __polyregion_gpu_subgroup_barrier();
    const auto isActive = [active](uint32_t source) { return source < 32u && (active & (1u << source)) != 0u; };
    const auto downSource = lane + 1u;
    const auto upSource = lane - 1u;
    const auto xorSource = lane ^ 1u;
    const auto expectedDown = isActive(downSource) ? downSource : lane;
    const auto expectedUp = lane > 0u && isActive(upSource) ? upSource : lane;
    const auto expectedXor = isActive(xorSource) ? xorSource : lane;
    const uint32_t failures = (size < 1u ? 1u : 0u) | (down != expectedDown ? 2u : 0u) | (up != expectedUp ? 4u : 0u)
                              | (index != (lane & ~clamp) ? 8u : 0u) | (xored != expectedXor ? 16u : 0u)
                              | (segmented != (lane & ~7u) ? 32u : 0u) | (masked != (lane < 16u ? 0u : lane) ? 64u : 0u)
                              | (ballot != 1u ? 128u : 0u) | (!any ? 256u : 0u) | (!all ? 512u : 0u);
    if (globalLane < 32u) result[globalLane] = failures;
  });
  bool valid = true;
  for (uint32_t lane = 0; lane < 32; ++lane)
    if (result[lane] != 0) {
      std::printf("%u:%u;", lane, result[lane]);
      valid = false;
    }
  if (valid) std::printf("42");
  return 0;
}
