#pragma region case: read
#pragma region using: size=1,2,10
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -o {output} {input}
#pragma region do: {output}
#pragma region requires: ok

#pragma region case: mutate
#pragma region using: size=1,2,10
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIZE_DEF={size} -DCHECK_MUT -o {output} {input}
#pragma region do: {output}
#pragma region requires: ok

#ifndef CHECK_SIZE_DEF
  #error "CHECK_SIZE_DEF undefined"
#endif

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

template <int N> struct MyArray {
  int elems[N];
  int other;
  int &operator[](size_t i) { return elems[i]; }
};

int main() {
  MyArray<CHECK_SIZE_DEF> m{};
  for (int i = 0; i < CHECK_SIZE_DEF; ++i)
    m[i] = i + 1;
  m.other = 7;

  int result = __polyregion_offload_f1__([m]() mutable {
#ifdef CHECK_MUT
    m[CHECK_SIZE_DEF - 1] *= 42;
#endif
    return m[CHECK_SIZE_DEF - 1] + m.other;
  });

#ifdef CHECK_MUT
  bool ok = result == CHECK_SIZE_DEF * 42 + 7;
#else
  bool ok = result == CHECK_SIZE_DEF + 7;
#endif
  printf("%s", ok ? "ok" : "bad");
  return 0;
}
