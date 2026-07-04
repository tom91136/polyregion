#pragma region case: factorial
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCASE=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3628800

#pragma region case: swap-even
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCASE=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 10

#pragma region case: swap-odd
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCASE=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 20

#ifndef CASE
  #define CASE 0
#endif

#include <cstdio>

#include "test_utils.h"

// tail recursion
static int fact(int n, int acc) { return n <= 1 ? acc : fact(n - 1, acc * n); }

// tail recursion with swapped args
static int swap_pick(int a, int b, int n) { return n <= 0 ? a : swap_pick(b, a, n - 1); }

int main() {
  int r = __polyregion_offload_f1__([=]() {
#if CASE == 0
    return fact(10, 1);
#elif CASE == 1
    return swap_pick(10, 20, 4);
#else
    return swap_pick(10, 20, 5);
#endif
  });
  printf("%d", r);
  return 0;
}
