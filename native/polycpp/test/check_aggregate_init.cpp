#pragma region case: aggregate_init
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdio>

#include "test_utils.h"

struct Inner {
  int a;
  int b;
};

struct Outer {
  Inner inner;
  int c;
  int d;
};

struct Box {
  int xs[3];
  int n;
};

int main() {
  const int result = __polyregion_offload_f1__([=]() {
    Inner flat{3, 4};
    Outer nested{Inner{5, 6}, 7, 8};
    Outer partial{Inner{9}};
    Outer blank{};
    Box box{{1, 2, 3}, 4};
    Box copy = box;
    int xs[4]{5, 6};

    const int flatSum = flat.a + flat.b * 10;
    const int nestedSum = nested.inner.a + nested.inner.b * 10 + nested.c * 100 + nested.d * 1000;
    const int partialSum = partial.inner.a + partial.inner.b * 10 + partial.c * 100 + partial.d * 1000;
    const int blankSum = blank.inner.a + blank.inner.b + blank.c + blank.d;
    const int boxSum = copy.xs[0] + copy.xs[1] * 10 + copy.xs[2] * 100 + copy.n * 1000;
    const int arrSum = xs[0] + xs[1] * 10 + xs[2] * 100 + xs[3] * 1000;
    return flatSum + nestedSum * 10 + partialSum * 100 + boxSum * 1000 + arrSum * 10000 + blankSum * 100000;
  });

  const bool ok = result == 5059593;
  std::printf(ok ? "pass" : "fail (result=%d)", result);
  return ok ? 0 : 1;
}
