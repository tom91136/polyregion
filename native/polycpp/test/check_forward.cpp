
// #CASE: capture
// #RUN: polycpp -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native   -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 42 43 44

// #CASE: capture-no-builtin
// #RUN: polycpp -O1 -g3 -fno-builtin-std-forward -fsanitize=address -fstdpar -fstdpar-arch=host@native -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 42 43 44

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct foo {
    int a, b, c;
  };
  foo value{42, 43, 44};
  foo c = __polyregion_offload_f1__([&]() { return std::forward<foo &&>(value); });
  printf("%d %d %d", c.a, c.b, c.c);
  return 0;
}
