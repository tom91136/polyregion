#pragma region case: =ptr_arith
#pragma region using: size=1,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -o {output} {input}
#pragma region do: {output}
#pragma region requires: -1 -1 1

#pragma region case: =ptr_arith=42
#pragma region using: size=1,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -DCHECK_MUT -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 42 1


#pragma region case: &ptr_arith
#pragma region using: size=1,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -o {output} {input}
#pragma region do: {output}
#pragma region requires: -1 -1 1


#pragma region case: &ptr_arith=42
#pragma region using: size=1,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -DCHECK_MUT -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 42 1

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
    *((xs + CHECK_SIZE_DEF) - 1) = 42;
#endif
    return *(xs + (CHECK_SIZE_DEF - 1));
  });

  printf("%d %d %d",
         result,                                                              //
         xs[CHECK_SIZE_DEF - 1],                                              //
         std::reduce(xs, xs + CHECK_SIZE_DEF - 1, 0) == -(CHECK_SIZE_DEF - 1) //
  );
  delete[] xs;
  return 0;
}
