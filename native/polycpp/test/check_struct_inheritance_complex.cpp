// #CASE: inheritance
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fstdpar -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: 0 3.000000 4.000000 1 2 5 6 7.000000

#include <cstddef>
#include <cstdio>

int main() {

  struct Base {
    int x;
  };

  struct A : Base {
    float x;
    float y;
  };

  struct B {
    int x;
    char y;
  };
  struct C : B, A {
    int a;
    int b;
    float c;
  };
  C value{{1, 2}, {{0}, 3, 4}, 5, 6, 7};

  C result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %f %f %d %d %d %d %f", result.A::Base::x, result.A::x, result.A::y, result.B::x, result.B::y, result.a, result.b, result.c);
  return 0;
}
