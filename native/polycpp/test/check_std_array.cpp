#pragma region case: =array
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -o {output} {input}
#pragma region do: {output}
#pragma region requires: -1 -1

#pragma region case: &std_array
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -o {output} {input}
#pragma region do: {output}
#pragma region requires: -1 -1

#pragma region case: =std_array*=42
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -DCHECK_MUT -o {output} {input}
#pragma region do: {output}
#pragma region requires: -42 -1

#pragma region case: &std_array*=42
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -DCHECK_MUT -o {output} {input}
#pragma region do: {output}
#pragma region requires: -42 -42

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

  std::array<int, CHECK_SIZE_DEF> xs = {};
  std::fill(xs.begin(), xs.end(), -1);
  int result = __polyregion_offload_f1__([CHECK_CAPTURE]() mutable {
#ifdef CHECK_MUT
    xs[CHECK_SIZE_DEF - 1] *= 42;
#endif
//    int aaa = xs[4];
//    return 42;
    return xs[CHECK_SIZE_DEF - 1];
  });
  printf("%d %d", result, xs[CHECK_SIZE_DEF - 1]);



  return 0;
}
