// #CASE: for
// #RUN: polycpp -fno-crash-diagnostics -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host LIMIT=10 {output}
//   #EXPECT: 10 1 2 3 4 5 6 7 8 9 10

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
