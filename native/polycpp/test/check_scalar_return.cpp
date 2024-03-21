
// #CASE: uint32_t
// #RUN: polycpp -fstdpar -fstdpar-arch=host@native -DCHECK_TYPE_DEF=uint32_t -DCHECK_TYPE_VAL=42 -DCHECK_TYPE_FMT="%d" -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 42

// #CASE: uint64_t
// #RUN: polycpp -fstdpar -fstdpar-arch=host@native -DCHECK_TYPE_DEF=uint64_t -DCHECK_TYPE_VAL=42 -DCHECK_TYPE_FMT="%ld" -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 42

// #CASE: float
// #RUN: polycpp -fstdpar -fstdpar-arch=host@native -DCHECK_TYPE_DEF=float -DCHECK_TYPE_VAL=0.42f -DCHECK_TYPE_FMT="%f" -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 0.420000

// #CASE: double
// #RUN: polycpp -fstdpar -fstdpar-arch=host@native -DCHECK_TYPE_DEF=double -DCHECK_TYPE_VAL=0.42 -DCHECK_TYPE_FMT="%f" -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 0.420000

#ifndef CHECK_TYPE_DEF
  #error "CHECK_TYPE_DEF undefined"
#endif

#ifndef CHECK_TYPE_VAL
  #error "CHECK_TYPE_VAL undefined"
#endif

#ifndef CHECK_TYPE_FMT
  #error "CHECK_TYPE_FMT undefined"
#endif

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {
  CHECK_TYPE_DEF c = __polyregion_offload_f1__([]() { return CHECK_TYPE_VAL; });
  printf(CHECK_TYPE_FMT, c);
  return 0;
}
