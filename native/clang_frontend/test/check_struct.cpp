// #CASE: capture
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fstdpar -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: 42

#include <cstddef>
#include <cstdio>


int main() {

  struct foo{int a;};
  foo value{42};
  foo c =  __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d", c.a);
  return 0;
}
