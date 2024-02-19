// #CASE: inheritance
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fstdpar -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: 11 22 33 44

#include <cstddef>
#include <cstdio>

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
