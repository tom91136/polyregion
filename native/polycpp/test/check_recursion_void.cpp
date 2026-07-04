#pragma region case: fill-sum
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 36

#include <cstdio>
#include <vector>

#include "test_utils.h"

static void fillrec(int *out, int n) {
  if (n <= 0) return;
  out[n - 1] = n;
  fillrec(out, n - 1); // recursion with effect
}

int main() {
  std::vector<int> v(8, 0);
  int *p = v.data();
  __polyregion_offload_f1__([=]() {
    fillrec(p, 8);
    return 0;
  });
  int s = 0;
  for (int x : v)
    s += x; // 1+2+...+8 = 36
  printf("%d", s);
  return 0;
}
