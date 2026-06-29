#pragma region case: string-const
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 209

#include <cstdio>

#include "test_utils.h"

int main() {
  const int r = __polyregion_offload_f1__([=]() {
    const char *s = "Xy";
    return static_cast<int>(s[0]) + static_cast<int>(s[1]); // 'X'(88) + 'y'(121)
  });
  printf("%d", r);
  return 0;
}
