#pragma region case: long-double
#pragma region offload-only
#pragma region compile-fails: Unsupported builtin type
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}

#pragma region case: member-pointer
#pragma region offload-only
#pragma region compile-fails: Unsupported type
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}

#include "test_utils.h"

struct S {
  int a;
};

int main() {
#if CHECK_KIND == 0
  long double d = 1.5L;
  const int r = __polyregion_offload_f1__([=]() { return static_cast<int>(d); });
#elif CHECK_KIND == 1
  int S::*mp = &S::a;
  S s{7};
  const int r = __polyregion_offload_f1__([=]() { return s.*mp; });
#endif
  return r;
}
