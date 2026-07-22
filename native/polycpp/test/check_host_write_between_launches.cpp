#pragma region case: host-write-between-launches
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 7

#include <cstdio>

#include "test_utils.h"

static void bump(int *x) {
  __polyregion_offload_f1__([=]() {
    x[0] += 1;
    return 0;
  });
}

int main() {
  int x[1] = {41};
  bump(x);
  std::printf("%d ", x[0]);

  x[0] = 6;
  bump(x);
  std::printf("%d", x[0]);
  return 0;
}
