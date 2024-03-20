// #CASE: =ptr
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -o {output} {input}
// #RUN: POLYSTL_PLATFORM=cuda {output}
//   #EXPECT: -1 -1 1

// #CASE: =ptr=42
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -DCHECK_MUT -o {output} {input}
// #RUN: POLYSTL_PLATFORM=hsa {output}
//   #EXPECT: 42 42 1


// #CASE: &ptr
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -o {output} {input}
// #RUN: POLYSTL_PLATFORM=hsa {output}
//   #EXPECT: -1 -1 1


// #CASE: &ptr=42
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -DCHECK_MUT -o {output} {input}
// #RUN: POLYSTL_PLATFORM=hsa {output}
//   #EXPECT: 42 42 1

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

  int *xs = new int[CHECK_SIZE_DEF];
  std::fill(xs, xs + CHECK_SIZE_DEF, -1);
  int result = __polyregion_offload_f1__([CHECK_CAPTURE]() {
#ifdef CHECK_MUT
    xs[CHECK_SIZE_DEF - 1] = 42;
#endif
    return xs[CHECK_SIZE_DEF - 1];
  });

  printf("%d %d %d",
         result,                                                              //
         xs[CHECK_SIZE_DEF - 1],                                              //
         std::reduce(xs, xs + CHECK_SIZE_DEF - 1, 0) == -(CHECK_SIZE_DEF - 1) //
  );
  delete[] xs;
  return 0;
}
