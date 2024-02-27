// #CASE: inheritance
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fstdpar -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: 0 3.000000 4.000000 12 42 2 33 120 121 5 6 7.000000

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct Base {
    int x;
  };

  struct Base2 {
    int x;
  };

  struct A : Base {
    float x;
    float y;
  };

  struct B : Base2 {
    int x;
    int y;
    int z;
  };
  struct X {
    int a, b;
  };
  struct C : X, B, A {
    int a;
    int b;
    float c;
  };
  C value{X{120, 121}, B{Base2{12}, 42, 2, 33}, A{Base{0}, 3, 4}, 5, 6, 7};

  C result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %f %f %d %d %d %d %d %d %d %d %f", //
         result.A::Base::x,                     //
         result.A::x,                           //
         result.A::y,                           //
         result.B::Base2::x,                    //
         result.B::x,                           //
         result.B::y,                           //
         result.B::z,                           //
         result.X::a,                           //
         result.X::b,                           //
         result.a, result.b, result.c);
  return 0;
}
