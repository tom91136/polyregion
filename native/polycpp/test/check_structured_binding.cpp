#pragma region case: array-ref
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_BINDING=1 -o {output} {input}
#pragma region do: X=3 {output}
#pragma region requires: 33

#pragma region case: aggregate
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_BINDING=2 -o {output} {input}
#pragma region do: X=3 {output}
#pragma region requires: 33

#pragma region case: tuple
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_BINDING=3 -o {output} {input}
#pragma region do: X=3 {output}
#pragma region requires: 33

#pragma region case: array-value
#pragma region offload-only
#pragma region compile-fails: Unsupported by-value array structured binding at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_BINDING=4 -o {output} {input}

#pragma region case: ref-writeback
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_BINDING=5 -o {output} {input}
#pragma region do: X=3 {output}
#pragma region requires: 105

#pragma region case: shadowed
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_BINDING=6 -o {output} {input}
#pragma region do: X=3 {output}
#pragma region requires: 39

#ifndef CHECK_BINDING
  #error "CHECK_BINDING undefined"
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include "test_utils.h"

struct Pair {
  int a;
  int b;
};

int main() {
  auto x = std::stoi(std::getenv("X"));

  int result = __polyregion_offload_f1__([=]() {
#if CHECK_BINDING == 1
    int xs[2] = {x, x * 10};
    auto &[a, b] = xs;
    return a + b;
#elif CHECK_BINDING == 2
    Pair p{x, x * 10};
    auto [a, b] = p;
    return a + b;
#elif CHECK_BINDING == 3
    std::pair<int, int> p{x, x * 10};
    auto [a, b] = p;
    return a + b;
#elif CHECK_BINDING == 4
    int xs[2] = {x, x * 10};
    auto [a, b] = xs;
    return a + b;
#elif CHECK_BINDING == 5
    int xs[2] = {x, x * 10};
    auto &[a, b] = xs;
    a = 100;
    b = 5;
    return xs[0] + xs[1];
#elif CHECK_BINDING == 6
    int total = 0;
    {
      Pair p{x, x * 10};
      auto [a, b] = p;
      total += a + b;
    }
    {
      std::pair<int, int> q{x, x};
      auto [a, b] = q;
      total += a + b;
    }
    return total;
#endif
  });
  std::printf("%d", result);
  return 0;
}
