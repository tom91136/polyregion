#pragma region case: =std_vector
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -o {output} {input}
#pragma region do: {output}
#pragma region requires: -1 -1

#pragma region case: &std_vector
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -o {output} {input}
#pragma region do: {output}
#pragma region requires: -1 -1

#pragma region case: =std_vector*=42
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -DCHECK_MUT -o {output} {input}
#pragma region do: {output}
#pragma region requires: -42 -1

#pragma region case: &std_vector*=42
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
