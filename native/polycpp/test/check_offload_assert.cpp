#pragma region case: assert-roundtrip
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 ASRT out of bounds

#include <cstdio>

#include "polyregion/enums.h"

#include "test_utils.h"

int main() {
  int r = __polyregion_offload_f1__([=]() {
    __polyregion_builtin_assert(static_cast<uint32_t>(polyregion::invoke::AssertCode::Assert), "out of bounds");
    return 0;
  });
  (void)r;
  const auto a = polyregion::polystl::details::lastAssert();
  const char cc[5] = {char(a.code), char(a.code >> 8), char(a.code >> 16), char(a.code >> 24), 0};
  printf("%d %s %s", a.raised ? 1 : 0, cc, a.message.c_str());
  return 0;
}
