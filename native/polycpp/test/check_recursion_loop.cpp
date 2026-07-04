#pragma region case: for-loop
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCASE=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 15

#pragma region case: while-loop
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCASE=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 15

#ifndef CASE
  #define CASE 0
#endif

#include <cstdio>

#include "test_utils.h"

static int fsum(int n) {
  if (n <= 0) return 0;
  int s = 0;

  for (int i = 0; i < n; i++)
    s += fsum(i); // recursive call inside a loop
  return s + n;
}

static int wsum(int n) {
  if (n <= 0) return 0;
  int s = 0, i = 0;
  while (i < n) {
    s += wsum(i);
    i++;
  }
  return s + n;
}

int main() {
  int r = __polyregion_offload_f1__([=]() {
#if CASE == 0
    return fsum(4);
#else
    return wsum(4);
#endif
  });
  printf("%d", r);
  return 0;
}
