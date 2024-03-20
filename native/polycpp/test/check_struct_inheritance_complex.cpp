// #CASE: inheritance
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fstdpar -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: 1 2 3 4.000000 5.000000 5 6 7.000000

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct Base {
    int x;
  };

  struct A : Base {
    float x;
    float y;
  };

  struct B : A {
    int x;
    char y;
  };
  struct C : B {
    int a;
    int b;
    float c;
  };
  C value{    //
          B{  //
            A{//
              Base{3}, 4, 5},
            1, 2},
          5, 6, 7};

  C result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf( //
      "%d %d "
      "%d "
      "%f %f "
      "%d %d %f",
      result.B::x, result.B::y,    //
      result.A::Base::x,           //
      result.A::x, result.A::y,    //
      result.a, result.b, result.c //
  );
  return 0;
}
