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

int main() {
  NestedAnon value{{7, 11}, 13};
  NestedAnonUnion other{{{19, 23}}, 29};
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
  const bool ok = result == 8148363;
  std::printf(ok ? "pass" : "fail (result=%d)", result);
  return ok ? 0 : 1;
}
