#pragma region case: assert-barrier
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 -1 10 lane 0 tripped

#include <cstdint>
#include <cstdio>

#include "polyregion/enums.h"

#include "test_utils.h"

int main() {
  int slot[2] = {-1, -1};
  int out[2] = {-1, -1};
  __polyregion_offload_workgroup__(2, [=, &slot, &out](uint32_t lid) {
    if (lid == 0) __polyregion_builtin_assert(static_cast<uint32_t>(polyregion::invoke::AssertCode::Assert), "lane 0 tripped");
    if (lid < 2) slot[lid] = static_cast<int>(lid) + 10;
    __polyregion_builtin_gpu_barrier_global();
    if (lid < 2) out[lid] = slot[1 - lid];
  });
  const auto a = polyregion::polystl::details::lastAssert();
  printf("%d %d %d %s", a.raised ? 1 : 0, out[0], out[1], a.message.c_str());
  return 0;
}
