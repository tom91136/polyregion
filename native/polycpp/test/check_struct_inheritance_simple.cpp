#pragma region case: inheritance
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 11 22 33 44

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct Base {
    int a, b;
  };

  struct Derived : Base {
    int c, d;
  };
  Derived value{{11, 22}, 33, 44};

  Derived result = __polyregion_offload_f1__([CHECK_CAPTURE]() {
    return value;
  });
  printf("%d %d %d %d", result.a, result.b, result.c, result.d);
  return 0;
}
