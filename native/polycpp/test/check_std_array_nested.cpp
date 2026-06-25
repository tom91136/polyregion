#pragma region case: std_array_nested
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 36

#include <array>
#include <cstdio>

#include "test_utils.h"

template <typename T> struct V4 {
  T x, y, z, w;
};

int main() {
  int r = __polyregion_offload_f1__([]() {
    std::array<std::array<V4<float>, 3>, 1> t = {};
    t[0][0].x = 6.0f;
    t[0][1].y = 12.0f;
    t[0][2].w = 18.0f;
    return static_cast<int>(t[0][0].x + t[0][1].y + t[0][2].w);
  });
  printf("%d", r);
  return 0;
}
