#pragma region case: fib-10
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DFIB_N=10 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 55

#pragma region case: fib-15
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DFIB_N=15 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 610

#pragma region case: fib-20
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DFIB_N=20 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 6765

#ifndef FIB_N
  #define FIB_N 10
#endif

#include <cstdio>

#include "test_utils.h"

static int fib(int n) {
  if (n < 2) return n;
  return fib(n - 1) + fib(n - 2); // binary recursion
}

int main() {
  int result = __polyregion_offload_f1__([=]() { return fib(FIB_N); });
  printf("%d", result);
  return 0;
}
