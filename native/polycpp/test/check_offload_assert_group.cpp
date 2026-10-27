#pragma region case: assert-wide-group
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 63 -1 lane 7 tripped

#include <cstdint>
#include <cstdio>

#include "polyregion/enums.h"

#include "test_utils.h"

int main() {
  constexpr uint32_t lanes = 64;
  int slot[lanes];
  int out[lanes];
  for (uint32_t i = 0; i < lanes; ++i)
    slot[i] = out[i] = -1;
  __polyregion_offload_workgroup__(lanes, [=, &slot, &out](uint32_t lid) {
    slot[lid] = static_cast<int>(lid) + 10;
    if (lid == 7) __polyregion_builtin_assert(static_cast<uint32_t>(polyregion::invoke::AssertCode::Assert), "lane 7 tripped");
    __polyregion_gpu_barrier_global();
    out[lid] = slot[(lid + 1) % lanes];
  });
  int correct = 0;
  for (uint32_t i = 0; i < lanes; ++i)
    if (out[i] == static_cast<int>((i + 1) % lanes) + 10) ++correct;
  const auto a = polyregion::polystl::details::lastAssert();
  printf("%d %d %d %s", a.raised ? 1 : 0, correct, out[7], a.message.c_str());
  return 0;
}
