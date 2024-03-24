// #CASE: capture
// #MATRIX: capture=&,=,value
// #RUN: polycpp -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 10 10

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct vec_impl {
    int *begin, *end;
  };
  struct vec {
    vec_impl impl;
  };

  int size = 10;

  int *xs = new int[size];
  for (int i = 0; i < size; ++i)
    xs[i] = i;

  vec value{vec_impl{xs, xs + size}};
  // == 1+9
  int result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return *(value.impl.begin + 1) + *(value.impl.end - 1); });
  printf("%d %d", result, *(value.impl.begin + 1) + *(value.impl.end - 1));
  delete[] xs;
  return 0;
}
