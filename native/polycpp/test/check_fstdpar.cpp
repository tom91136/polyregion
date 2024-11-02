
// #CASE: with
// #RUN: polycpp -fno-crash-diagnostics -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: OK

// #CASE: without
// #RUN: polycpp -fno-crash-diagnostics -O1 -g3 -fsanitize=address -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
//   #EXPECT: OK

#include <cstdio>

int main() {
  printf("OK");
  return 0;
}
