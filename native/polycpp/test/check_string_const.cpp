#pragma region case: string-const
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 209

// both cases pin the sign of plain `char` explicitly: it is signed on x86 but unsigned on arm/ppc/s390x
#pragma region case: string-const-direct
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fsigned-char -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 338

#pragma region case: string-const-direct-unsigned-char
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -funsigned-char -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 594

#include <cstdio>

#include "test_utils.h"

#if CHECK_KIND == 1
static int funcHead() { return static_cast<int>(__func__[0]); } // 'f'(102)
#endif

int main() {
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() {
    const char *s = "Xy";
    return static_cast<int>(s[0]) + static_cast<int>(s[1]); // 'X'(88) + 'y'(121)
  });
#elif CHECK_KIND == 1
  const int r = __polyregion_offload_f1__([=]() {
    const char *s = "Xy";
    const int direct = static_cast<int>("Xy"[0]);     // 'X'(88)
    const int high = static_cast<int>("\303\251"[0]); // 0xc3 reads -61 signed, 195 unsigned
    return direct + high + funcHead() + static_cast<int>(s[0]) + static_cast<int>(s[1]);
  });
#endif
  printf("%d", r);
  return 0;
}
