#pragma region case: try-inside-loop
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 101 102 103 4 105 106 107 108 3

#pragma region case: throw-exits-loop
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 101 102 103 4 0 0 0 0 3

#pragma region case: throw-exits-nested-loop
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 11 12 13 14 5 0 0 0 4

#pragma region case: never-thrown
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 11 12 13 14 15 16 17 18 0

#pragma region case: throw-in-nested-if
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 2 4 6 8 5 0 0 0 4

#pragma region case: resume-after-catch
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 2 3 0 54 55 56 57 2

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
      try {
        p[i] = i + 1;
        if (i == 3) throw i;
        p[i] += 100;
      } catch (int e) {
        p[8] = e;
      }
    }
    return 0;
  });
#elif CHECK_KIND == 1
  __polyregion_offload_f1__([=]() {
    try {
      for (int i = 0; i < 8; i++) {
        p[i] = i + 1;
        if (i == 3) throw i;
        p[i] += 100;
      }
    } catch (int e) {
      p[8] = e;
    }
    return 0;
  });
#elif CHECK_KIND == 2
  __polyregion_offload_f1__([=]() {
    try {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          int k = i * 3 + j;
          p[k] = k + 1;
          if (i == 1 && j == 1) throw k;
          p[k] += 10;
        }
      }
    } catch (int e) {
      p[8] = e;
    }
    return 0;
  });
#elif CHECK_KIND == 3
  __polyregion_offload_f1__([=]() {
    try {
      for (int i = 0; i < 8; i++) {
        p[i] = i + 1;
        if (i == 99) throw i;
        p[i] += 10;
      }
    } catch (int e) {
      p[8] = e;
    }
    return 0;
  });
#elif CHECK_KIND == 4
  __polyregion_offload_f1__([=]() {
    try {
      for (int i = 0; i < 8; i++) {
        p[i] = i + 1;
        if (i >= 2) {
          if (i == 4) throw i;
        }
        p[i] *= 2;
      }
    } catch (int e) {
      p[8] = e;
    }
    return 0;
  });
#elif CHECK_KIND == 5
  __polyregion_offload_f1__([=]() {
    try {
      for (int i = 0; i < 4; i++) {
        p[i] = i + 1;
        if (i == 2) throw i;
      }
    } catch (int e) {
      p[8] = e;
    }
    for (int i = 4; i < 8; i++)
      p[i] = 50 + i;
    return 0;
  });
#else
  #error "CHECK_KIND undefined"
#endif
  for (int i = 0; i < 9; i++)
    std::printf("%d%s", data[i], i == 8 ? "" : " ");
  return 0;
}
