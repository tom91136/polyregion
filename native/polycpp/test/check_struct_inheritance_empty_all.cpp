// #CASE: inheritance
// #MATRIX: capture=&,=,value
// #RUN: polycpp -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 0 0

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct Base {};

  struct Derived : Base {};
  Derived value{};

  static_assert(sizeof(Derived) == 1);

  char valueBytes[1], actualBytes[1];
  memcpy(valueBytes, &value, 1);

  Derived result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });

  memcpy(actualBytes, &value, 1);
  printf("%d %d", valueBytes[0], actualBytes[0]);
  return 0;
}
