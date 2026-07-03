#pragma region case: break-in-cond
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 10

#pragma region case: continue-in-cond
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 25

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdio>

#include "test_utils.h"

int main() {
  int n = 10;
  int result = __polyregion_offload_f1__([=]() {
    int s = 0;
#if CHECK_KIND == 0
    for (int i = 0; i < n; i++) {
      if (i == 5) break;
      s += i;
    }
#else
    int i = 0;
    while (i < n) {
      int cur = i;
      i = i + 1;
      if (cur % 2 == 0) continue;
      s += cur;
    }
#endif
    return s;
  });
  printf("%d", result);
  return 0;
}
