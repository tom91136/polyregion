#pragma region case: uncaught-scalar
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 [EXCP] [int] 0

#pragma region case: uncaught-struct
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 [EXCP] [Err] 0

#pragma region case: handler-type-mismatch
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 [EXCP] [Err] 0

#pragma region case: never-raised
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 [] [] 5

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdio>

#include "polyregion/enums.h"

#include "test_utils.h"

struct Err {
  int code;
  int mul;
};

int main() {
  int data[4] = {1, 0, 3, 4};
  int *p = data;
#if CHECK_KIND == 0
  __polyregion_offload_f1__([=]() {
    if (p[0] == 1) throw p[2];
    p[1] = 5;
    return 0;
  });
#elif CHECK_KIND == 1
  __polyregion_offload_f1__([=]() {
    if (p[0] == 1) throw Err{p[2], p[3]};
    p[1] = 5;
    return 0;
  });
#elif CHECK_KIND == 2
  __polyregion_offload_f1__([=]() {
    try {
      if (p[0] == 1) throw Err{p[2], p[3]};
    } catch (int e) {
      p[1] = e;
    }
    p[1] = 5;
    return 0;
  });
#elif CHECK_KIND == 3
  __polyregion_offload_f1__([=]() {
    if (p[0] == 99) throw p[2];
    p[1] = 5;
    return 0;
  });
#else
  #error "CHECK_KIND undefined"
#endif
  const auto a = polyregion::polystl::details::lastAssert();
  const char cc[5] = {char(a.code), char(a.code >> 8), char(a.code >> 16), char(a.code >> 24), 0};
  std::printf("%d [%s] [%s] %d", a.raised ? 1 : 0, cc, a.message.c_str(), data[1]);
  return 0;
}
