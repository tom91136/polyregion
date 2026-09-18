#pragma region case: conditional_reference_lifetime
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7 9 7 9

#include <cstdio>

#include "test_utils.h"

namespace {

constexpr int Left = 7;
constexpr int Right = 9;

const int &choose(bool left) { return left ? Left : Right; }

} // namespace

int main() {
  int *out = new int[4];
  int *conditions = new int[2]{1, 0};
  __polyregion_offload_f1__([=]() {
    out[0] = choose(conditions[0] != 0);
    out[1] = choose(conditions[1] != 0);
    const int &left = conditions[0] != 0 ? Left : Right;
    const int &right = conditions[1] != 0 ? Left : Right;
    out[2] = left;
    out[3] = right;
    return 0;
  });
  std::printf("%d %d %d %d", out[0], out[1], out[2], out[3]);
  delete[] conditions;
  delete[] out;
  return 0;
}
