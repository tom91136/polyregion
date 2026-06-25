#pragma region case: enum-compare
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: SEL=1 {output}
#pragma region requires: 42
#pragma region do: SEL=2 {output}
#pragma region requires: 7

#include <cstdio>
#include <cstdlib>

#include "test_utils.h"

enum geom { g_rect = 1, g_circ = 2 };

int main() {
  int sel = std::atoi(std::getenv("SEL"));
  int r = __polyregion_offload_f1__([=]() {
    if (sel == g_rect) return 42;
    if (sel == g_circ) return 7;
    return -1;
  });
  printf("%d", r);
  return 0;
}
