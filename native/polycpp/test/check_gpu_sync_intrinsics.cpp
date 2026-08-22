#pragma region case: gpu-sync-intrinsics
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -Werror -Wno-error=unused-command-line-argument -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdint>
#include <cstdio>

#include "test_utils.h"

int main() {
  if (!polyregionTestOffloadEnabled()) {
    __polyregion_gpu_fence_global();
    __polyregion_gpu_fence_local();
    __polyregion_gpu_fence_all();
    __polyregion_gpu_barrier_global();
    __polyregion_gpu_barrier_local();
    __polyregion_gpu_barrier_all();
    __polyregion_gpu_subgroup_barrier();
    std::printf("42");
    return 0;
  }

  uint32_t storage[65]{};
  auto *storagePtr = storage;
  __polyregion_offload_workgroup__(32, [storagePtr](const uint32_t lane) {
    [[clang::annotate("__polyregion_local")]] uint32_t localBarrier[32];
    [[clang::annotate("__polyregion_local")]] uint32_t localAll[32];

    if (lane < 32u) storagePtr[lane] = lane + 1u;
    __polyregion_gpu_fence_global();
    __polyregion_gpu_barrier_global();

    if (lane < 32u) localBarrier[lane] = lane + 1u;
    __polyregion_gpu_fence_local();
    __polyregion_gpu_barrier_local();

    if (lane < 32u) {
      storagePtr[32u + lane] = lane + 1u;
      localAll[lane] = lane + 1u;
    }
    __polyregion_gpu_fence_all();
    __polyregion_gpu_barrier_all();

    // Subgroup barrier has no portable cross-subgroup observation; keep it on the uniform path as a codegen smoke.
    __polyregion_gpu_subgroup_barrier();

    if (lane == 0) {
      uint32_t globalBarrierSum = 0;
      uint32_t localBarrierSum = 0;
      uint32_t globalAllSum = 0;
      uint32_t localAllSum = 0;
      for (uint32_t i = 0; i < 32u; ++i) {
        globalBarrierSum += storagePtr[i];
        localBarrierSum += localBarrier[i];
        globalAllSum += storagePtr[32u + i];
        localAllSum += localAll[i];
      }
      if (globalBarrierSum == 528u && localBarrierSum == 528u && globalAllSum == 528u && localAllSum == 528u) storagePtr[64] = 42;
    }
  });
  std::printf("%u", storage[64]);
  return 0;
}
