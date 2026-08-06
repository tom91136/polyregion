#pragma region case: throw-escapes-barrier-loop
#pragma region offload-only
#pragma region compile-fails: collective barrier
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}

#pragma region case: catch-inside-barrier-loop
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 2 2

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdint>
#include <cstdio>

#include "test_utils.h"

int main() {
#if CHECK_KIND == 0
  int data[4] = {-1, -1, -1, -1};
  int *p = data;
  const int r = __polyregion_offload_f1__([=]() {
    for (int i = 0; i < 4; i++) {
      if (i == 2) throw i;
      p[i] = i;
      __polyregion_builtin_gpu_barrier_global();
    }
    return 0;
  });
  std::printf("%d %d %d %d %d", r, data[0], data[1], data[2], data[3]);
#elif CHECK_KIND == 1
  int caught[2] = {-1, -1};
  int *p = caught;
  __polyregion_offload_workgroup__(2, [=](uint32_t lid) {
    int hits = 0;
    for (int i = 0; i < 4; i++) {
      try {
        if (i == 2) throw i;
      } catch (int e) {
        hits = e;
      }
      __polyregion_builtin_gpu_barrier_global();
    }
    p[lid] = hits;
  });
  std::printf("%d %d", caught[0], caught[1]);
#else
  #error "CHECK_KIND undefined"
#endif
  return 0;
}
