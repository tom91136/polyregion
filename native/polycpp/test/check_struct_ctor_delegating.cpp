#pragma region case: ctor_delegating
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdio>

#include "test_utils.h"

struct S {
  int v;
  S() : S(7) {}
  S(int v) : v(v) {}
};

int main() {
  int seed = 11;
  const int result = __polyregion_offload_f1__([=]() {
    S a;
    S b(seed);
    return a.v * 100 + b.v;
  });
  const bool ok = result == 711;
  std::printf(ok ? "pass" : "fail (result=%d)", result);
  return ok ? 0 : 1;
}
