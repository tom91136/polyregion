#pragma region case: =multiregion
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: -1 -2 -1 -2

#include <algorithm>
#include <cstdio>

#include "test_utils.h"

int main() {
  int *xs = new int[8];
  long *ys = new long[8];
  std::fill(xs, xs + 8, -1);
  std::fill(ys, ys + 8, -2);

  int a = __polyregion_offload_f1__([xs]() { return xs[7]; });
  long b = __polyregion_offload_f1__([ys]() { return ys[7]; });

  printf("%d %ld %d %d", a, b, xs[7], static_cast<int>(ys[7]));
  delete[] xs;
  delete[] ys;
  return 0;
}
