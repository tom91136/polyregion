#pragma region case: capture
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 43 44

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct foo{int a,b,c;};
  foo value{42, 43, 44};
  foo c =  __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %d %d", c.a, c.b, c.c);
  return 0;
}
