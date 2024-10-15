// #CASE: capture
// #MATRIX: capture=&,=,value
// #RUN: polycpp -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 42 43 44 45

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct bar {};

  struct foo {
    int a, b, c;
    bar bar;
    int d;
  };
  foo value{42, 43, 44, bar{}, 45};
  foo c = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %d %d %d", c.a, c.b, c.c, c.d);
  return 0;
}
