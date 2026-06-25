#pragma region case: ref-var-by-copy
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 18

#include <cstdio>

#include "test_utils.h"

struct Buf {
  int *data;
  int &operator[](int i) const { return data[i]; }
};

struct Fields {
  Buf a;
  Buf b;
};

int main() {
  int da[4] = {0, 0, 0, 0};
  int db[4] = {0, 0, 0, 0};
  Fields fields{Buf{da}, Buf{db}};
  Fields &f = fields;
  int r = __polyregion_offload_f1__([=]() {
    f.a[0] = 7;
    f.b[0] = 11;
    return f.a[0] + f.b[0]; // 18
  });
  printf("%d", r);
  return 0;
}
