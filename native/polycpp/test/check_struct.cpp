// #CASE: capture
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fstdpar -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: 42 43 44

#include <cstddef>
#include <cstdio>


int main() {

  struct foo{int a,b,c;};
  foo value{42, 43, 44};
  foo c =  __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %d %d", c.a, c.b, c.c);
  return 0;
}