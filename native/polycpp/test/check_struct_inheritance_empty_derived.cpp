#pragma region case: inheritance
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 33 44

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct Base {
    int c, d;
  };

  struct Derived : Base {};
  Derived value{33, 44};

  static_assert(sizeof(Derived) == sizeof(int) * 2);

  Derived result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %d", result.c, result.d);
  return 0;
}
