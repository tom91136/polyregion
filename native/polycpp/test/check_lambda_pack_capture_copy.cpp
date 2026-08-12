#pragma region case: pack-capture-copy
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 20

#include <cstdio>

#include "test_utils.h"

template <typename... Ts> auto sum(Ts... values) {
  return [values...]() { return (values + ...); };
}

int main() {
  const int result = __polyregion_offload_f1__([=]() {
    auto packed = sum(9, 11);
    auto packedCopy = packed;
    return packedCopy();
  });
  std::printf("%d", result);
  return 0;
}
