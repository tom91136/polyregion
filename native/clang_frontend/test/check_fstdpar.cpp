
// #CASE: with
// #RUN: polycpp -fstdpar -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: OK

// #CASE: without
// #RUN: polycpp -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: OK

#include <cstdio>

int main() {
  printf("OK");
  return 0;
}
