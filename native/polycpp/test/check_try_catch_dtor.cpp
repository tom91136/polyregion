#pragma region case: dtor-before-handler
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 701 1

#pragma region case: reverse-order
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 21 21

#pragma region case: outer-object-survives
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 19

#pragma region case: loop-scope-dtor
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 111 111

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdio>

#include "test_utils.h"

struct Trace {
  int *sink;
  int id;
  ~Trace() { sink[0] = sink[0] * 10 + id; }
};

int main() {
  int *sink = new int[2]{0, 0};
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    try {
      Trace t{sink, 1};
      if (sink[1] == 0) throw 7;
    } catch (int e) {
      v = e * 100 + sink[0];
    }
    return v;
  });
#elif CHECK_KIND == 1
  const int r = __polyregion_offload_f1__([=]() {
    try {
      Trace a{sink, 1};
      Trace b{sink, 2};
      if (sink[1] == 0) throw 0;
    } catch (...) {
    }
    return sink[0];
  });
#elif CHECK_KIND == 2
  const int r = __polyregion_offload_f1__([=]() {
    int v = 0;
    Trace outer{sink, 9};
    try {
      Trace inner{sink, 1};
      if (sink[1] == 0) throw 0;
    } catch (...) {
      v = sink[0];
    }
    return v;
  });
#elif CHECK_KIND == 3
  const int r = __polyregion_offload_f1__([=]() {
    try {
      for (int i = 0; i < 5; i++) {
        Trace t{sink, 1};
        if (i == 2) throw i;
      }
    } catch (int e) {
      (void)e;
    }
    return sink[0];
  });
#else
  #error "CHECK_KIND undefined"
#endif
  std::printf("%d %d", r, sink[0]);
  delete[] sink;
  return 0;
}
