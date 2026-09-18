#pragma region case: builtin-launder
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input} -g0
#pragma region do: {output}
#pragma region requires: 42 44 45 12 12 9

#include <cstdio>
#include <new>

#include "test_utils.h"

namespace {

struct Pair {
  int left;
  int right;
};

} // namespace

int main() {
  int captured = 42;
  auto *outside = std::launder(&captured);
  int capturedRaw = 44;
  auto *raw = &capturedRaw;
  auto *out = new int[6];

  __polyregion_offload_f1__([=]() {
    out[0] = *__builtin_launder(outside);
    out[1] = *std::launder(raw);

    int local = 43;
    auto *localPtr = std::launder(&local);
    *localPtr += 2;
    out[2] = *localPtr;

    Pair pair{5, 7};
    auto *pairPtr = __builtin_launder(&pair);
    pairPtr->left += pairPtr->right;
    out[3] = pairPtr->left;

    Pair subobject{4, 12};
    auto *memberPtr = std::launder(&subobject.right);
    out[4] = *memberPtr;

    int nested = 8;
    auto *once = std::launder(&nested);
    auto *twice = __builtin_launder(once);
    *twice += 1;
    out[5] = *once;
    return 0;
  });

  std::printf("%d %d %d %d %d %d", out[0], out[1], out[2], out[3], out[4], out[5]);
  delete[] out;
  return 0;
}
