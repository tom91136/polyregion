#pragma region case: for
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: LIMIT=10 {output}
#pragma region requires: 10 1 2 3 4 5 6 7 8 9 10

#include <cstdio>
#include <string>
#include <vector>

#include "test_utils.h"

int main() {
  auto limit = std::stoi(std::getenv("LIMIT"));
  int *xs = new int[limit];
  std::fill(xs, xs + limit, -1);
  int result = __polyregion_offload_f1__([&]() {
    for (int i = 0; i < limit; ++i) {
      xs[i] = i + 1;
    }
    return limit;
  });
  printf("%d", result);
  for (int i = 0; i < limit; ++i) {
    printf(" %d", xs[i]);
  }
  delete[] xs;
  return 0;
}
