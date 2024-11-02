// #CASE: capture
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fno-crash-diagnostics -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 42 43 44

// #CASE: capture-no-builtin
// #MATRIX: capture=&,=,value flag=-fno-builtin-std-forward
// #RUN: polycpp -fno-crash-diagnostics -O1 -g3 {flag} -fsanitize=address -fstdpar -fstdpar-arch=host@native -DCHECK_CAPTURE={capture} -o {output} {input}
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
  foo c = __polyregion_offload_f1__([CHECK_CAPTURE]() { return std::move(value); });
  printf("%d %d %d", c.a, c.b, c.c);
  return 0;
}
