// #CASE: =std_vector
// #MATRIX: size=1,2,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: -1 -1

// #CASE: &std_vector
// #MATRIX: size=1,2,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: -1 -1

// #CASE: =std_vector*=42
// #MATRIX: size=1,2,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -DCHECK_MUT -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: -42 -1

// #CASE: &std_vector*=42
// #MATRIX: size=1,2,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -DCHECK_MUT -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: -42 -42

#ifndef CHECK_SIZE_DEF
  #error "CHECK_SIZE_DEF undefined"
#endif

#ifndef CHECK_CAPTURE
  #error "CHECK_CAPTURE undefined"
#endif

#include <cstddef>
#include <cstdio>
#include <cstring>

#include "test_utils.h"

int main() {

  std::vector<int> xs(CHECK_SIZE_DEF, -1);
  int result = __polyregion_offload_f1__([CHECK_CAPTURE]() mutable {
#ifdef CHECK_MUT
    xs[CHECK_SIZE_DEF - 1] *= 42;
#endif
    return xs[CHECK_SIZE_DEF - 1];
  });
  printf("%d %d", result, xs[CHECK_SIZE_DEF - 1]);



  return 0;
}
