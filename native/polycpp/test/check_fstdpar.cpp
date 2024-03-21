
// #CASE: with
// #RUN: polycpp -fstdpar -fstdpar-arch=host@native -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: OK

// #CASE: without
// #RUN: polycpp -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: OK

#include <cstdio>

int main() {
  printf("OK");
  return 0;
}
