#pragma region case: scope-exit
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7 42

#pragma region case: reverse-order
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 21

#pragma region case: nested-scope
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 42

#pragma region case: loop-continue
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 111 111

#pragma region case: early-return
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 42

#pragma region case: loop-break
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 111 111

#pragma region case: member-dtor-effects
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=6 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 42

#include <cstdio>

#include "test_utils.h"

struct Guard {
  int *sink;
  ~Guard() { sink[0] = 42; }
};

struct Trace {
  int *sink;
  int id;
  ~Trace() { sink[0] = sink[0] * 10 + id; }
};

// empty body, but destroying the member is not a no-op
struct Outer {
  Guard inner;
  ~Outer() {}
};

int main() {
  int *sink = new int[2]{0, 0};
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() {
    Guard g{sink};
    return 7;
  });
#elif CHECK_KIND == 1
  const int r = __polyregion_offload_f1__([=]() {
    Trace a{sink, 1};
    Trace b{sink, 2};
    return 0;
  });
#elif CHECK_KIND == 2
  const int r = __polyregion_offload_f1__([=]() {
    {
      Guard g{sink};
    }
    return sink[0];
  });
#elif CHECK_KIND == 3
  const int r = __polyregion_offload_f1__([=]() {
    int i = 0;
    while (i < 3) {
      Trace t{sink, 1};
      ++i;
      if (i < 3) continue;
    }
    return sink[0];
  });
#elif CHECK_KIND == 4
  const int r = __polyregion_offload_f1__([=]() {
    Guard g{sink};
    if (sink[1] == 0) return sink[0];
    return 9;
  });
#elif CHECK_KIND == 5
  const int r = __polyregion_offload_f1__([=]() {
    for (int i = 0; i < 5; ++i) {
      Trace t{sink, 1};
      if (i == 2) break;
    }
    return sink[0];
  });
#elif CHECK_KIND == 6
  const int r = __polyregion_offload_f1__([=]() {
    Outer o{{sink}};
    return 0;
  });
#else
  #error "CHECK_KIND undefined"
#endif
  std::printf("%d %d", r, sink[0]);
  delete[] sink;
  return 0;
}
