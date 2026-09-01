#pragma region case: local_array_decay
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 41 43

#include <cstdio>

#include "test_utils.h"

int main() {
  int *out = new int[2];
  __polyregion_offload_f1__([=]() {
    int values[4] = {40, 41, 42, 43};
    int *assigned = nullptr;
    assigned = values;
    int *offset = values + 3;
    out[0] = assigned[1];
    out[1] = *offset;
    return 0;
  });
  std::printf("%d %d", out[0], out[1]);
  delete[] out;
  return 0;
}
