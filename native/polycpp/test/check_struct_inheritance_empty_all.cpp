#pragma region case: inheritance
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct Base {};

  struct Derived : Base {};
  Derived value{};

  static_assert(sizeof(Base) == 1);
  static_assert(sizeof(Derived) == 1);

  char valueBytes[1], actualBytes[1];
  memcpy(valueBytes, &value, 1);

  Derived result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });

  memcpy(actualBytes, &value, 1);
  printf("%d %d", valueBytes[0], actualBytes[0]);
  return 0;
}
