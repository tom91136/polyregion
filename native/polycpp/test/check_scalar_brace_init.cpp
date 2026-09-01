#pragma region case: scalar-brace-init
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 85

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {
  const int result = __polyregion_offload_f1__([=]() {
    std::size_t zero{0};
    std::size_t value{42};
    int scalar{};
    return int(zero + value) + scalar + 43;
  });
  std::printf("%d", result);
  return 0;
}
