#pragma region case: class-iter
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1234

#pragma region case: class-iter-ref
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 30

#pragma region case: class-iter-nested
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3702

#pragma region case: array
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 5678

#ifndef CHECK_KIND
  #error "CHECK_KIND undefined"
#endif

#include <cstdio>

#include "test_utils.h"

struct Span {
  struct Iter {
    int *p;
    int i;
    int &operator*() const { return p[i]; }
    Iter &operator++() {
      ++i;
      return *this;
    }
    bool operator!=(const Iter &that) const { return i != that.i; }
  };
  int *data;
  int n;
  Iter begin() const { return Iter{data, 0}; }
  Iter end() const { return Iter{data, n}; }
};

int main() {
  int *xs = new int[4]{1, 2, 3, 4};
  const Span s{xs, 4};
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() {
    int acc = 0;
    for (int v : s)
      acc = acc * 10 + v;
    return acc;
  });
#elif CHECK_KIND == 1
  const int r = __polyregion_offload_f1__([=]() {
    for (int &v : s)
      v *= 3;
    int acc = 0;
    for (int v : s)
      acc += v;
    return acc;
  });
#elif CHECK_KIND == 2
  const int r = __polyregion_offload_f1__([=]() {
    int acc = 0;
    for (int a : s) {
      int row = 0;
      for (int b : s) {
        if (b == 3) break;
        row += b;
      }
      acc = acc * 10 + a * row;
    }
    return acc;
  });
#elif CHECK_KIND == 3
  const int r = __polyregion_offload_f1__([=]() {
    int ys[4] = {5, 6, 7, 8};
    int acc = 0;
    for (const int v : ys)
      acc = acc * 10 + v;
    return acc;
  });
#endif
  std::printf("%d", r);
  delete[] xs;
  return 0;
}
