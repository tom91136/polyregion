#pragma region case: ptr_incdec
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 10 30 30 10 1

#include <cstdio>

#include "test_utils.h"

int main() {
  int *xs = new int[4];
  for (int i = 0; i < 4; ++i)
    xs[i] = (i + 1) * 10;
  int *out = new int[5];

  __polyregion_offload_f1__([=]() {
    int *p = xs;
    out[0] = *p++;
    out[1] = *++p;
    out[2] = *p--;
    out[3] = *--p;
    out[4] = p == xs ? 1 : 0;
    return 0;
  });

  std::printf("%d %d %d %d %d", out[0], out[1], out[2], out[3], out[4]);
  delete[] out;
  delete[] xs;
  return 0;
}
