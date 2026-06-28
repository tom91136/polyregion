#pragma region case: bitfields
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdio>

#include "test_utils.h"

struct PackedBits {
  unsigned lo : 3;
  unsigned mid : 5;
  unsigned hi : 8;
};

struct SignedBits {
  int low : 4;
  int high : 6;
};

int main() {
  PackedBits bits{1, 17, 203};
  SignedBits signedBits{-3, 17};
  const int result = __polyregion_offload_f1__([=]() mutable {
    bits.lo = 5;
    bits.mid = bits.mid + 3;
    signedBits.high = -11;
    return static_cast<int>(bits.lo) * 1000000 + static_cast<int>(bits.mid) * 10000 + static_cast<int>(bits.hi) * 100 +
           static_cast<int>(signedBits.low + 30) * 10 + static_cast<int>(signedBits.high + 30);
  });
  const bool ok = result == 5220589;
  std::printf(ok ? "pass" : "fail (result=%d)", result);
  return ok ? 0 : 1;
}
