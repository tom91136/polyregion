// #CASE: capture
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fstdpar -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: 42 43 44 45

#include <cstddef>
#include <cstdio>

int main() {

  struct bar {
    int d;
  };
  struct foo {
    int a, b, c;
    bar bar;
  };
  foo value{42, 43, 44, bar{45}};
  foo c = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %d %d %d", c.a, c.b, c.c, c.bar.d);
  return 0;
}
