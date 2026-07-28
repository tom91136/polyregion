#pragma region case: value-capture
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_LAMBDA=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#pragma region case: template-arg
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_LAMBDA=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 83

#pragma region case: generic
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_LAMBDA=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 84

#pragma region case: ref-capture
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_LAMBDA=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 43

#pragma region case: aggregate-capture
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_LAMBDA=5 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 91

#pragma region case: template-instantiation
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_LAMBDA=6 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 111

#include <cstdio>

#include "test_utils.h"

struct Pair {
  int a;
  int b;
};

template <typename F> int twice(const F &f) { return f(1) + f(2); }

template <typename T> int scaled(T seed) {
  auto value = [seed]() { return static_cast<int>(seed) + 1; };
  return value();
}

int main() {
  int base = 40;
  int result = __polyregion_offload_f1__([=]() mutable {
#if CHECK_LAMBDA == 1
    auto add = [base](int x) { return base + x; };
    return add(2);
#elif CHECK_LAMBDA == 2
    return twice([base](int x) { return base + x; });
#elif CHECK_LAMBDA == 3
    auto add = [base](auto x) { return base + x; };
    return add(2) + static_cast<int>(add(2.5f));
#elif CHECK_LAMBDA == 4
    auto accumulate = [&base](int x) { base += x; };
    accumulate(1);
    accumulate(2);
    return base;
#elif CHECK_LAMBDA == 5
    int xs[4] = {1, 2, 3, 4};
    Pair p{base / 4, base / 2};
    auto sum = [xs, p](int k) {
      int acc = p.a + p.b;
      for (int i = 0; i < 4; ++i)
        acc += xs[i] * k;
      return acc;
    };
    auto one = []() { return 1; };
    return twice(sum) + one();
#elif CHECK_LAMBDA == 6
    return scaled(base / 4) + scaled(0.5f) * 100;
#else
  #error "CHECK_LAMBDA undefined"
#endif
  });
  std::printf("%d", result);
  return 0;
}
