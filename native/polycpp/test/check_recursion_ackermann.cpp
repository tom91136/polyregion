#pragma region case: ack-2-3
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCASE=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 9

#pragma region case: ack-3-3
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCASE=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 61

#ifndef CASE
  #define CASE 0
#endif

#include <cstdio>

#include "test_utils.h"

static int ack(int m, int n) {
  if (m == 0) return n + 1;
  if (n == 0) return ack(m - 1, 1);
  return ack(m - 1, ack(m, n - 1));
}

int main() {
  int r = __polyregion_offload_f1__([=]() {
#if CASE == 0
    return ack(2, 3);
#else
    return ack(3, 3);
#endif
  });
  printf("%d", r);
  return 0;
}
