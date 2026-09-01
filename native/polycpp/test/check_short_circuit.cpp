#pragma region case: short_circuit
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 2 0 1 1 1

#include <cstdio>

#include "test_utils.h"

int main() {
  int *out = new int[5];
  int *conditions = new int[4]{0, 1, 1, 0};
  __polyregion_offload_f1__([=]() {
    int sideEffects = 0;
    const bool falseAnd = conditions[0] != 0 && (++sideEffects == 1);
    const bool trueOr = conditions[1] != 0 || (++sideEffects == 1);
    const bool trueAnd = conditions[2] != 0 && (++sideEffects == 1);
    const bool falseOr = conditions[3] != 0 || (++sideEffects == 2);
    out[0] = sideEffects;
    out[1] = falseAnd;
    out[2] = trueOr;
    out[3] = trueAnd;
    out[4] = falseOr;
    return 0;
  });
  std::printf("%d %d %d %d %d", out[0], out[1], out[2], out[3], out[4]);
  delete[] conditions;
  delete[] out;
  return 0;
}
