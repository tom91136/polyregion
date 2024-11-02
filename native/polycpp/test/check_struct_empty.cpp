// #CASE: capture
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fno-crash-diagnostics -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct foo{};
  foo value{};
  __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });

  return 0;
}
