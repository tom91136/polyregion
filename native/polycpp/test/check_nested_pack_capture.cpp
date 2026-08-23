#pragma region case: nested pack capture
#pragma region using: capture=&,=
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 90

#include <cstdio>

#include "test_utils.h"

struct Queue {
  template <typename F> decltype(auto) submit(F f) const {
    int handler = 0;
    return f(handler);
  }
};

struct Submitter {
  template <typename... Ts> int operator()(Ts... values) const {
    // Clang captures each expanded value, but folds this non-ODR-used constant without a field.
    const int factor = 3;
    return Queue{}.submit([&](int &) { return (values + ...) * factor; });
  }
};

int main() {
  int result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return Submitter{}(10, 20); });
  std::printf("%d", result);
  return 0;
}
