#pragma region case: anon_structs
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdio>

#include "test_utils.h"

struct NestedAnon {
  struct {
    int x;
    int y;
  };
  int z;
};

struct NestedAnonUnion {
  union {
    struct {
      int x;
      int y;
    };
    long long raw;
  };
  int z;
};

struct AnonWithCtor {
  struct {
    int x;
    int y;
  };
  int z;
  AnonWithCtor(int x, int z) : x(x), z(z) { y = x * 2; }
};

struct TwoAnon {
  struct {
    int a;
  };
  struct {
    int b;
  };
  int c;
};

int main() {
  NestedAnon value{{7, 11}, 13};
  NestedAnonUnion other{{{19, 23}}, 29};
  TwoAnon two{{1}, {2}, 3};
  const int result = __polyregion_offload_f1__([=]() mutable {
    NestedAnon copy = value;
    copy.x += 1;
    copy.y += 2;
    copy.z += 3;
    NestedAnonUnion otherCopy{{{1, 2}}, 3};
    otherCopy = other;
    otherCopy.x += 4;
    otherCopy.y += 5;
    otherCopy.z += 6;
    return copy.x * 1000000 + copy.y * 10000 + copy.z * 1000 + otherCopy.x * 100 + otherCopy.y + otherCopy.z;
  });
  const int extra = __polyregion_offload_f1__([=]() {
    AnonWithCtor built(3, 5);
    TwoAnon copy = two;
    return built.x * 100000 + built.y * 10000 + built.z * 1000 + copy.a * 100 + copy.b * 10 + copy.c;
  });
  const bool ok = result == 8148363 && extra == 365123;
  std::printf(ok ? "pass" : "fail (result=%d extra=%d)", result, extra);
  return ok ? 0 : 1;
}
