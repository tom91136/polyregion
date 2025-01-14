#pragma region case: =std_list
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -o {output} {input}
#pragma region do: {output}
#pragma region requires: -1 -1

#pragma region case: &std_list
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -o {output} {input}
#pragma region do: {output}
#pragma region requires: -1 -1

#pragma region case: =std_list*=42
#pragma region using: size=1,2,10,100
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -DCHECK_MUT -o {output} {input}
#pragma region do: {output}
#pragma region requires: -42 -1

#pragma region case: &std_list*=42
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
#include <list>

#include "test_utils.h"

int main() {


  std::list<int> xs(CHECK_SIZE_DEF, -1);
  int result = __polyregion_offload_f1__([CHECK_CAPTURE]() mutable {
  #ifdef CHECK_MUT
      auto it = xs.begin();
      std::advance(it, CHECK_SIZE_DEF - 1);
      *it *= 42;
  #endif
      auto it = xs.begin();
      std::advance(it, CHECK_SIZE_DEF - 1);
      return *it;
  });

  auto it = xs.begin();
  std::advance(it, CHECK_SIZE_DEF - 1);
  printf("%d %d", result, *it);


  return 0;
}
