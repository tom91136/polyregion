
#pragma region case: uint32_t
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_TYPE_DEF=uint32_t -DCHECK_TYPE_VAL=42 -DCHECK_TYPE_FMT="%d" -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#pragma region case: uint64_t
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_TYPE_DEF=uint64_t -DCHECK_TYPE_VAL=42 -DCHECK_TYPE_FMT="%ld" -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#pragma region case: float
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_TYPE_DEF=float -DCHECK_TYPE_VAL=0.42f -DCHECK_TYPE_FMT="%f" -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0.420000

#pragma region case: double
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_TYPE_DEF=double -DCHECK_TYPE_VAL=0.42 -DCHECK_TYPE_FMT="%f" -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0.420000

#ifndef CHECK_TYPE_DEF
  #error "CHECK_TYPE_DEF undefined"
#endif

#ifndef CHECK_TYPE_VAL
  #error "CHECK_TYPE_VAL undefined"
#endif

#ifndef CHECK_TYPE_FMT
  #error "CHECK_TYPE_FMT undefined"
#endif

#ifndef CHECK_CAPTURE
  #error "CHECK_CAPTURE undefined"
#endif

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {
  CHECK_TYPE_DEF value = CHECK_TYPE_VAL;
  CHECK_TYPE_DEF c = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf(CHECK_TYPE_FMT, c);
  return 0;
}
