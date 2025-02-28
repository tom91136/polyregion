#pragma region case: capture
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 43 44 45 46 47 48 49 50 51 52 53 54

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct bar2 {
    int f;
    int g;
  };

  struct baz {
    int e;
    bar2 bar2;
  };

  struct bar {
    int d;
    baz baz;
  };

  struct foo {
    int a, b, c;
    bar bar;
    int d;
    baz baz;
    bar2 bar2;
  };

  foo value{42, 43, 44, bar{45, baz{46, bar2{47, 48}}}, 49, baz{50, bar2{51, 52}}, bar2{53, 54}};
  foo c = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %d %d %d %d %d %d %d %d %d %d %d %d", //
         c.a, c.b, c.c,                            //
         c.bar.d,                                  //
         c.bar.baz.e,                              //
         c.bar.baz.bar2.f,                         //
         c.bar.baz.bar2.g,                         //
         c.d,                                      //
         c.baz.e,                                  //
         c.baz.bar2.f,                             //
         c.baz.bar2.g,                             //
         c.bar2.f,                                 //
         c.bar2.g);
  return 0;
}
