// #CASE: inheritance
// #MATRIX: capture=&,=,value
// #RUN: polycpp -O1 -g3 -fsanitize=address -fstdpar -fstdpar-arch=host@native -DCHECK_CAPTURE={capture} -o {output} {input}
// #RUN: POLYSTL_PLATFORM=host {output}
// #EXPECT: 12 3.000000 42 5.000000 7.000000

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct Base {
    int x;
  };

  struct A : Base {
    float y;
  };

  struct B : Base {
    float z;
  };

  struct C : A, B {
    float w;
  };

  C value{A{Base{12}, 3.0f}, B{Base{42}, 5.0f}, 7.0f};

  C result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  printf("%d %f %d %f %f",             //
         static_cast<A&>(result).x,            // From A's Base
         result.y,                     // From A's y
         static_cast<B&>(result).x,            // From B's Base
         result.z,                     // From B's z
         result.w);                    // From C's w
  return 0;
}
