#pragma region case: null
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#pragma region case: distinct
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0

#pragma region case: same
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#pragma region case: order
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#include <cstdio>

#include "test_utils.h"

int main() {
  int xs[2] = {1, 1}; // equal elements so distinct only passes if addresses are compared
#if CHECK_KIND == 0
  const int *nil = nullptr;
  const int *a = &xs[0];
  const int r = __polyregion_offload_f1__([=]() { return nil == nullptr && a != nullptr ? 1 : 0; });
#elif CHECK_KIND == 1
  const int *a = &xs[0];
  const int *b = &xs[1];
  const int r = __polyregion_offload_f1__([=]() { return a == b ? 1 : 0; });
#elif CHECK_KIND == 2
  const int *a = &xs[0];
  const int *b = xs;
  const int r = __polyregion_offload_f1__([=]() { return a == b ? 1 : 0; });
#elif CHECK_KIND == 3
  const int *a = &xs[0];
  const int *b = &xs[1];
  const int r = __polyregion_offload_f1__([=]() { return a < b && b > a && a <= b && b >= a ? 1 : 0; });
#endif
  std::printf("%d", r);
  return 0;
}
