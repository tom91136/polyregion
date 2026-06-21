#pragma region case: =ptr_array
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE== -o {output} {input}
#pragma region do: {output}
#pragma region requires: 33 11

#pragma region case: &ptr_array
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE=& -o {output} {input}
#pragma region do: {output}
#pragma region requires: 33 11

#pragma region case: =ptr_array=42
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE== -DCHECK_MUT -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 42

#pragma region case: &ptr_array=42
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE=& -DCHECK_MUT -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 42

#ifndef CHECK_CAPTURE
  #error "CHECK_CAPTURE undefined"
#endif

#include <cstdio>

#include "test_utils.h"

static int run(int *const ptrs[3]) {
  return __polyregion_offload_f1__([CHECK_CAPTURE]() {
#ifdef CHECK_MUT
    ptrs[1][1] = 42;
    return ptrs[1][1];
#else
    return ptrs[0][0] + ptrs[1][1] + ptrs[2][2];
#endif
  });
}

int main() {

  int d0[3] = {0, 1, 2};
  int d1[3] = {10, 11, 12};
  int d2[3] = {20, 21, 22};
  int *arr[3] = {d0, d1, d2};

  int result = run(arr);

  printf("%d %d", result, arr[1][1]);
  return 0;
}
