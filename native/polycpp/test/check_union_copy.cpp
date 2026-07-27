#pragma region case: union_copy
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdio>

#include "test_utils.h"

union Scalar {
  int i;
  float f;
};

union Buffer {
  int head;
  int slots[4];
};

struct Holder {
  Scalar s;
  Buffer b;
  int tag;
};

int main() {
  Holder value{};
  value.s.i = 5;
  value.b.slots[0] = 1;
  value.b.slots[1] = 2;
  value.b.slots[2] = 3;
  value.b.slots[3] = 4;
  value.tag = 6;

  Scalar punned{};
  punned.f = 2.5f;

  const int result = __polyregion_offload_f1__([=]() mutable {
    Holder copy = value;
    Scalar s2 = copy.s;
    Buffer b2 = copy.b;
    Buffer b3{};
    b3 = b2;
    Scalar s3 = static_cast<Scalar &&>(s2);
    Buffer b4 = static_cast<Buffer &&>(b3);
    Buffer b5{};
    b5 = static_cast<Buffer &&>(b4);
    Scalar p2 = punned;
    Scalar p3 = static_cast<Scalar &&>(p2);
    return s3.i + b2.slots[0] * 10 + b5.slots[3] * 100 + copy.tag * 1000 + (p3.f == 2.5f ? 10000 : 0);
  });

  const bool ok = result == 16415;
  std::printf(ok ? "pass" : "fail (result=%d)", result);
  return ok ? 0 : 1;
}
