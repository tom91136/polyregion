// #CASE: =ptr
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -fstdpar-arch=host@native -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: -1 -1 -1 == -1 -1 -1

// #CASE: =ptr=42,43,44
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -fstdpar-arch=host@native -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -DCHECK_MUT -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 42 43 44 == 42 43 44

// #CASE: &ptr
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -fstdpar-arch=host@native -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: -1 -1 -1 == -1 -1 -1

// #CASE: &ptr=42,43,44
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -fstdpar-arch=host@native -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -DCHECK_MUT -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 42 43 44 == 42 43 44

#ifndef CHECK_SIZE_DEF
  #error "CHECK_SIZE_DEF undefined"
#endif

#ifndef CHECK_CAPTURE
  #error "CHECK_CAPTURE undefined"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <numeric>

#include "test_utils.h"

int main() {

  struct Foo {
    int a, b, c;
  };
  Foo *xs = new Foo[CHECK_SIZE_DEF];
  std::fill(xs, xs + CHECK_SIZE_DEF, Foo{-1, -1, -1});
  Foo result = __polyregion_offload_f1__([CHECK_CAPTURE]() {
#ifdef CHECK_MUT
    auto &x = xs[CHECK_SIZE_DEF - 1];
    auto &m = x;
    m.a = 42;
    m.b = 43;
    m.c = 44;
#endif
    return xs[CHECK_SIZE_DEF - 1];
  });

  printf("%d %d %d == %d %d %d",
         result.a,                 //
         result.b,                 //
         result.c,                 //
         xs[CHECK_SIZE_DEF - 1].a, //
         xs[CHECK_SIZE_DEF - 1].b, //
         xs[CHECK_SIZE_DEF - 1].c  //
  );
  delete[] xs;
  return 0;
}
