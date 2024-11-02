// #CASE: inheritance
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fno-crash-diagnostics -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 33 44

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct Base {
    int c, d;
  };

  struct Derived : Base {};
  Derived value{33, 44};

  static_assert(sizeof(Derived) == sizeof(int) * 2);

  Derived result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %d", result.c, result.d);
  return 0;
}
