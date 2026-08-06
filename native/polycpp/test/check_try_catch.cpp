#pragma region case: no-throw
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 5

#pragma region case: catch-all
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7

#pragma region case: catch-exact-scalar
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 30

#pragma region case: catch-struct-payload
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 12

#pragma region case: first-handler-wins
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 103

#pragma region case: tail-after-try-runs
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 37

#pragma region case: nested-inner-catches
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=6 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 12

#pragma region case: throw-across-call
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=7 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 8

#pragma region case: throw-from-catch-body
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=8 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 55

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdio>

#include "test_utils.h"

struct Err {
  int code;
  int mul;
};

static int thrower(const int *p) {
  if (p[0] == 1) throw p[3];
  return 0;
}

int main() {
  int data[4] = {1, 2, 3, 4};
  int *p = data;
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      v = 5;
    } catch (...) {
      v = 99;
    }
    return v;
  });
#elif CHECK_KIND == 1
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw 1;
      v = 5;
    } catch (...) {
      v = 7;
    }
    return v;
  });
#elif CHECK_KIND == 2
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw p[2];
      v = 5;
    } catch (int e) {
      v = e * 10;
    }
    return v;
  });
#elif CHECK_KIND == 3
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw Err{p[2], p[3]};
      v = 5;
    } catch (const Err &e) {
      v = e.code * e.mul;
    }
    return v;
  });
#elif CHECK_KIND == 4
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw p[2];
      v = 5;
    } catch (int e) {
      v = 100 + e;
    } catch (...) {
      v = 200;
    }
    return v;
  });
#elif CHECK_KIND == 5
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw 1;
      v = 5;
    } catch (...) {
      v = 7;
    }
    v += 30;
    return v;
  });
#elif CHECK_KIND == 6
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      try {
        if (p[0] == 1) throw 1;
        v = 1;
      } catch (...) {
        v = 2;
      }
      v += 10;
    } catch (...) {
      v = 99;
    }
    return v;
  });
#elif CHECK_KIND == 7
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      v = thrower(p);
    } catch (int e) {
      v = e * 2;
    }
    return v;
  });
#elif CHECK_KIND == 8
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      try {
        if (p[0] == 1) throw p[1];
        v = 1;
      } catch (int e) {
        throw Err{e, p[3]};
      }
    } catch (const Err &e) {
      v = e.code * e.mul + 47;
    }
    return v;
  });
#else
  #error "CHECK_KIND undefined"
#endif
  std::printf("%d", r);
  return 0;
}
