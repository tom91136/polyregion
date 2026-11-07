#pragma region case: interior_alias
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "test_utils.h"

constexpr int K = 4;
constexpr uint32_t L = 32;
constexpr int W = static_cast<int>(L) + 2 * K;

#define RUN(tag, base_expr)                                                                                                                \
  do {                                                                                                                                     \
    int *base = (base_expr);                                                                                                               \
    int *p = base + K;                                                                                                                     \
    for (int i = 0; i < W; ++i)                                                                                                            \
      base[i] = i;                                                                                                                         \
    int seen[L];                                                                                                                           \
    __polyregion_offload_workgroup__(L, [=, &seen](uint32_t g) {                                                                           \
      p[g] += 1000;                                                                                                                        \
      seen[g] = base[K + g];                                                                                                               \
    });                                                                                                                                    \
    for (uint32_t g = 0; g < L; ++g) {                                                                                                     \
      bad += seen[g] != static_cast<int>(K + g) + 1000;                                                                                    \
      bad += base[K + g] != static_cast<int>(K + g) + 1000;                                                                                \
    }                                                                                                                                      \
    for (int i = 0; i < K; ++i)                                                                                                            \
      bad += base[i] != i;                                                                                                                 \
    __polyregion_offload_workgroup__(L, [=](uint32_t g) { base[g] = p[static_cast<int>(g) - K] + 1; });                                    \
    for (uint32_t g = 0; g < L; ++g)                                                                                                       \
      bad += base[g] != (g < K ? static_cast<int>(g) : static_cast<int>(g - K) + K + 1000) + 1;                                            \
    if (bad) std::printf("%s: %d mismatches\n", tag, bad);                                                                                 \
    total += bad;                                                                                                                          \
    bad = 0;                                                                                                                               \
  } while (0)

int main() {
  int bad = 0, total = 0;
  int *heap_new = new int[W];
  int *heap_malloc = static_cast<int *>(std::malloc(sizeof(int) * W));
  int stack[W];

  RUN("heap-new", heap_new);
  RUN("heap-malloc", heap_malloc);
  RUN("stack", stack);

  delete[] heap_new;
  std::free(heap_malloc);
  std::printf(total == 0 ? "pass" : "fail");
  return total == 0 ? 0 : 1;
}
