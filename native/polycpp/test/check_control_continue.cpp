#pragma region case: for-continue
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 4

#pragma region case: while-continue
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: range-for-continue
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 4

#pragma region case: nested-continue
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 15

// Stmt::Cont only targets the innermost loop and a switch lowers to one, so continuing the real loop from
// inside a switch needs multi-level continue that polyAST cannot currently express
#pragma region case: switch-continue
#pragma region offload-only
#pragma region compile-fails: Unsupported continue targeting an enclosing loop from a switch
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}

#include <cstdio>

#include "test_utils.h"

int main() {
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() {
    int acc = 0;
    for (int i = 0; i < 4; ++i) {
      if (i == 2) continue;
      acc += i; // 0+1+3
    }
    return acc;
  });
#elif CHECK_KIND == 1
  // continuing on the last iteration must still refresh the condition, else the loop runs once more
  const int r = __polyregion_offload_f1__([=]() {
    int acc = 0, i = 0;
    while (i < 3) {
      ++i;
      if (i == 3) continue;
      acc += i; // 1+2
    }
    return acc;
  });
#elif CHECK_KIND == 2
  int xs[4] = {0, 1, 2, 3};
  const int r = __polyregion_offload_f1__([=]() {
    int acc = 0;
    for (const int v : xs) {
      if (v == 2) continue;
      acc += v;
    }
    return acc;
  });
#elif CHECK_KIND == 3
  const int r = __polyregion_offload_f1__([=]() {
    int acc = 0;
    for (int i = 0; i < 4; ++i) {
      if (i == 1) continue;
      for (int j = 0; j < 4; ++j) {
        if (j == 3) continue;
        acc += i * j; // (0+2+3)*(0+1+2)
      }
    }
    return acc;
  });
#elif CHECK_KIND == 4
  const int r = __polyregion_offload_f1__([=]() {
    int acc = 0;
    for (int i = 0; i < 4; ++i) {
      switch (i) {
        case 1: continue;
        case 2: acc += 100; break;
        default: acc += 1; break;
      }
    }
    return acc;
  });
#endif
  std::printf("%d", r);
  return 0;
}
