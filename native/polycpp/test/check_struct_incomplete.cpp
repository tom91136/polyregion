#pragma region case: incomplete
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdio>

#include "test_utils.h"

struct Fwd;

struct Holder {
  Fwd *p;
  int v;
};

int main() {
  int backing = 5;
  Holder opaque{reinterpret_cast<Fwd *>(&backing), 42};
  Holder empty{nullptr, 7};
  const int result = __polyregion_offload_f1__([=]() {
    Holder a = opaque;
    Holder b = empty;
    return a.v * 100 + b.v;
  });
  const bool ok = result == 4207;
  std::printf(ok ? "pass" : "fail (result=%d)", result);
  return ok ? 0 : 1;
}
