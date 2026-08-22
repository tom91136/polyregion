#pragma region case: gpu-intrinsics-api
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdint>
#include <cstdio>

#include "polystl/intrinsics.h"

#include "test_utils.h"

int exercise() {
  uint32_t atomic = 10;
  uint32_t old = __polyregion_gpu_atomic_xchg_u32(&atomic, 4u);
  old += __polyregion_gpu_atomic_add_u32(&atomic, 3u);
  old += __polyregion_gpu_atomic_sub_u32(&atomic, 2u);
  old += __polyregion_gpu_atomic_min_u32(&atomic, 3u);
  old += __polyregion_gpu_atomic_max_u32(&atomic, 9u);
  old += __polyregion_gpu_atomic_and_u32(&atomic, 7u);
  old += __polyregion_gpu_atomic_or_u32(&atomic, 8u);
  old += __polyregion_gpu_atomic_xor_u32(&atomic, 3u);

  uint32_t scalar = 11;
  const auto loaded = __polyregion_gpu_volatile_load_u32(&scalar);
  __polyregion_gpu_volatile_store_u32(&scalar, loaded + 1u);

  __polyregion_gpu_barrier_global();
  __polyregion_gpu_barrier_local();
  __polyregion_gpu_barrier_all();
  __polyregion_gpu_fence_global();
  __polyregion_gpu_fence_local();
  __polyregion_gpu_fence_all();
  __polyregion_gpu_subgroup_barrier();

  const bool valid = __polyregion_gpu_global_idx(0) == 0 && __polyregion_gpu_global_size(0) == 1 && __polyregion_gpu_group_idx(0) == 0
                     && __polyregion_gpu_group_size(0) == 1 && __polyregion_gpu_local_idx(0) == 0 && __polyregion_gpu_local_size(0) == 1
                     && __polyregion_gpu_lane_idx() == 0 && __polyregion_gpu_subgroup_size() == 1
                     && __polyregion_gpu_shuffle_down_u32(7u, 1u, 31u) == 7 && __polyregion_gpu_shuffle_up_u32(7u, 1u, 31u) == 7
                     && __polyregion_gpu_shuffle_idx_u32(7u, 0u, 31u) == 7 && __polyregion_gpu_shuffle_xor_u32(7u, 1u, 31u) == 7
                     && __polyregion_gpu_ballot(UINT32_MAX, true) == 1 && __polyregion_gpu_vote_any(UINT32_MAX, true)
                     && __polyregion_gpu_vote_all(UINT32_MAX, true) && old == 48 && atomic == 10 && scalar == 12;
  return valid ? 42 : 0;
}

int main() {
  std::printf("%d", exercise());
  return 0;
}
