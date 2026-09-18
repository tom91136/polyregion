#pragma region case: early_return_inline
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 0 1 0

#include <cstdio>

#include "test_utils.h"

static bool different_from_previous(const int *values, int index) {
  if (index == 0) return true;
  return values[index] != values[index - 1];
}

int main() {
  int *values = new int[4]{0, 0, 1, 1};
  int *out = new int[4];

  __polyregion_offload_f1__([=]() {
    for (int i = 0; i < 4; ++i)
      out[i] = different_from_previous(values, i);
    return 0;
  });

  std::printf("%d %d %d %d", out[0], out[1], out[2], out[3]);
  delete[] out;
  delete[] values;
  return 0;
}
