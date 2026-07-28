#pragma region case: null-stmt
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: N=5 {output}
#pragma region requires: 10

#include <cstdio>
#include <cstdlib>
#include <string>

#include "test_utils.h"

int main() {
  auto n = std::stoi(std::getenv("N"));
  int result = __polyregion_offload_f1__([=]() {
    int sum = 0;
    ; // null-stmt
    for (int i = 0; i < n; ++i)
      sum += i;
    for (int i = 0; i < n; ++i)
      ; // null-stmt
    return sum;
  });
  std::printf("%d", result);
  return 0;
}
