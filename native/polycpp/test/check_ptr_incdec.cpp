#pragma region case: ptr_incdec
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 10 30 30 10 1 30 40 1 20 30 40

#include <cstdio>

#include "test_utils.h"

static int *advance_reference(int *&pointer, int by) {
  pointer += by;
  return pointer;
}

static int *bind_pointer_prvalue(int *&&pointer) { return pointer; }

static int &reference_at(int *pointer, int index) { return pointer[index]; }

int main() {
  int *xs = new int[4];
  for (int i = 0; i < 4; ++i)
    xs[i] = (i + 1) * 10;
  int *out = new int[11];

  __polyregion_offload_f1__([=]() {
    int *p = xs;
    out[0] = *p++;
    out[1] = *++p;
    out[2] = *p--;
    out[3] = *--p;
    out[4] = p == xs ? 1 : 0;
    out[5] = *advance_reference(p, 2);
    out[6] = *(++p);
    out[7] = p == xs + 3 ? 1 : 0;
    out[8] = *bind_pointer_prvalue(xs + 1);
    out[9] = *bind_pointer_prvalue(xs + 2);
    out[10] = *(&reference_at(xs, 3));
    return 0;
  });

  std::printf("%d %d %d %d %d %d %d %d %d %d %d", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10]);
  delete[] out;
  delete[] xs;
  return 0;
}
