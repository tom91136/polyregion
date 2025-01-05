#pragma region case: inheritance
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 43 44 45 46 47

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct baz {
    int e;
  };

  struct bar : baz {
    int d;
    baz baz;
  };
  struct foo {
    int a, b, c;
    bar bar;
  };
  foo value{42, 43, 44, bar{{45},46, baz{47}}};
  foo c = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %d %d %d %d %d", c.a, c.b, c.c, c.bar.e, c.bar.d, c.bar.baz.e);
  return 0;
}
