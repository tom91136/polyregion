#pragma region case: interior_below_base
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: interior_below_base_global
#pragma region targets: host cuda
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_GLOBAL=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "test_utils.h"

constexpr int K = 4;
constexpr uint32_t L = 64;
constexpr int W = static_cast<int>(L) + 2 * K;

static int global_array[W];

#define RUN(tag, base_expr)                                                                                                                \
  do {                                                                                                                                     \
    const int *base = (base_expr);                                                                                                         \
    const int *p = base + K;                                                                                                               \
    int d1[L], dk[L], all[L];                                                                                                              \
    __polyregion_offload_workgroup__(L, [=, &d1](uint32_t g) { d1[g] = p[-1] + p[g]; });                                                   \
    __polyregion_offload_workgroup__(L, [=, &dk](uint32_t g) { dk[g] = p[-K] + p[g]; });                                                   \
    __polyregion_offload_workgroup__(L, [=, &all](uint32_t g) { all[g] = p[static_cast<int>(g) - K]; });                                   \
    for (uint32_t g = 0; g < L; ++g) {                                                                                                     \
      bad += d1[g] != base[K - 1] + base[K + g];                                                                                           \
      bad += dk[g] != base[0] + base[K + g];                                                                                               \
      bad += all[g] != base[g];                                                                                                            \
    }                                                                                                                                      \
    if (bad) std::printf("%s: %d mismatches\n", tag, bad);                                                                                 \
    total += bad;                                                                                                                          \
    bad = 0;                                                                                                                               \
  } while (0)

int main() {
  int bad = 0, total = 0;
  for (int i = 0; i < W; ++i)
    global_array[i] = i + 100;

  int *heap_new = new int[W];
  int *heap_malloc = static_cast<int *>(std::malloc(sizeof(int) * W));
  int stack[W];
  for (int i = 0; i < W; ++i) {
    heap_new[i] = i + 100;
    heap_malloc[i] = i + 100;
    stack[i] = i + 100;
  }

#ifdef CHECK_GLOBAL
  RUN("global", global_array);
#else
  RUN("heap-new", heap_new);
  RUN("heap-malloc", heap_malloc);
  RUN("stack", stack);
#endif

  delete[] heap_new;
  std::free(heap_malloc);
  std::printf(total == 0 ? "pass" : "fail");
  return total == 0 ? 0 : 1;
}
