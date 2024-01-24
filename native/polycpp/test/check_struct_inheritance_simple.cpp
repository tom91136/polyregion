// #CASE: inheritance
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fstdpar -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: 1 2 3 4

#include <cstddef>
#include <cstdio>

int main() {

  struct A {
    int a, b;
  };

  struct B : A {
    int c, d;
  };
  B value{{1, 2}, 3, 4};

  B result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %d %d %d", result.a, result.b, result.c, result.d);
  return 0;
}
