#pragma region case: loop-partial
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 101 102 103 4 0 0 0 0 0 | 1 99 [loop]

#pragma region case: nested-if
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 2 4 6 8 5 0 0 0 0 | 1 5 [nested if]

#pragma region case: nested-loop
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 11 12 13 14 5 0 0 0 0 | 1 8 [inner]

#pragma region case: never-fired
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 11 12 13 14 15 16 17 18 0 | 0 0 []

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdio>

#include "test_utils.h"

int main() {
  int data[9] = {0};
  int *p = data;
#if CHECK_KIND == 0
  __polyregion_offload_f1__([=]() {
    for (int i = 0; i < 8; i++) {
      p[i] = i + 1;
      if (i == 3) __polyregion_builtin_assert(99, "loop");
      p[i] += 100;
    }
    return 0;
  });
#elif CHECK_KIND == 1
  __polyregion_offload_f1__([=]() {
    for (int i = 0; i < 8; i++) {
      p[i] = i + 1;
      if (i >= 2) {
        if (i == 4) __polyregion_builtin_assert(5, "nested if");
      }
      p[i] *= 2;
    }
    return 0;
  });
#elif CHECK_KIND == 2
  __polyregion_offload_f1__([=]() {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        int k = i * 3 + j;
        p[k] = k + 1;
        if (i == 1 && j == 1) __polyregion_builtin_assert(8, "inner");
        p[k] += 10;
      }
    }
    return 0;
  });
#elif CHECK_KIND == 3
  __polyregion_offload_f1__([=]() {
    for (int i = 0; i < 8; i++) {
      p[i] = i + 1;
      if (i == 99) __polyregion_builtin_assert(1, "never");
      p[i] += 10;
    }
    return 0;
  });
#endif
  const auto a = polyregion::polystl::details::lastAssert();
  for (int i = 0; i < 9; i++)
    printf("%d ", data[i]);
  printf("| %d %u [%s]", a.raised ? 1 : 0, a.code, a.message.c_str());
  return 0;
}
