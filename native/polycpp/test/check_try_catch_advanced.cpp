#pragma region case: catch-by-base
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: rethrow
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: throw-in-recursion
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3

#pragma region case: non-trivial-payload
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 37

#pragma region case: non-trivial-catch-all-is-rejected
#pragma region offload-only
#pragma region compile-fails: nearest matching handler must have the exact type
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdio>

#include "test_utils.h"

struct Base {
  int code;
};
struct Mid : Base {};
struct Derived : Mid {};
struct NonTrivial {
  int *sink;
  int code;
  ~NonTrivial() { *sink = *sink * 10 + 7; }
};

static int descend(const int *p, int n) {
  if (n == 0) throw p[2];
  return descend(p, n - 1) + 1;
}

int main() {
  int data[4] = {1, 0, 3, 4};
  int *p = data;
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw Derived{{{p[2]}}};
    } catch (const Base &e) {
      v = e.code;
    } catch (const Derived &) {
      v = 9;
    }
    return v;
  });
#elif CHECK_KIND == 1
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      try {
        if (p[0] == 1) throw p[2];
      } catch (int) {
        throw;
      }
    } catch (int e) {
      v = e;
    }
    return v;
  });
#elif CHECK_KIND == 2
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      v = descend(p, p[3]);
    } catch (int e) {
      v = e;
    }
    return v;
  });
#elif CHECK_KIND == 3
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      if (p[0] == 1) throw NonTrivial{p + 1, p[2]};
    } catch (const NonTrivial &e) {
      v = *e.sink * 10 + e.code;
    }
    return v * 10 + p[1];
  });
#elif CHECK_KIND == 4
  const int r = __polyregion_offload_f1__([=]() {
    try {
      if (p[0] == 1) throw NonTrivial{p + 1, p[2]};
    } catch (...) {
      return p[2];
    }
    return 0;
  });
#else
  #error "CHECK_KIND undefined"
#endif
  std::printf("%d", r);
  return 0;
}
