#pragma region case: even-10
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DEVEN_N=10 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#pragma region case: even-7
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DEVEN_N=7 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0

#ifndef EVEN_N
  #define EVEN_N 10
#endif

#include <cstdio>

#include "test_utils.h"

static bool is_odd(int n);
static bool is_even(int n) {
  if (n == 0) return true;
  return is_odd(n - 1); // mutual recursion
}
static bool is_odd(int n) {
  if (n == 0) return false;
  return is_even(n - 1); // mutual recursion
}

int main() {
  int r = __polyregion_offload_f1__([=]() { return is_even(EVEN_N) ? 1 : 0; });
  printf("%d", r);
  return 0;
}
