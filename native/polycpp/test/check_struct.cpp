// #CASE: capture
// #MATRIX: capture=&,=,value
// #RUN: polycpp -fstdpar -fstdpar-arch=cuda@sm_60 -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLYSTL_PLATFORM=cuda {output}
//   #EXPECT: 42 43 44

#include <cstddef>
#include <cstdio>

//#include "test_utils.h"

int main() {

  struct foo{int a,b,c;};
  foo value{42, 43, 44};
  foo c =   ([CHECK_CAPTURE]() { return value; })();
  printf("%d %d %d", c.a, c.b, c.c);
  return 0;
}
