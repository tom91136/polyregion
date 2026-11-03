#pragma region case: string-temp-record
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3072 6 1
#pragma region stderr-count: 0 failed to release

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <execution>
#include <string>
#include <vector>

#include "polyregion/enums.h"
#include "polystl/polystl.h"

#include "../test_utils.h"
#include "reflect-rt/rt_reflect.hpp"

int main() {
  constexpr std::size_t n = 1024;
  std::vector<std::size_t> out(n);
  std::size_t total = 0;
  for (int round = 0; round < 3; ++round) {
    total += std::string(static_cast<std::size_t>(round) + 1, 'a').size();
    std::for_each(std::execution::par_unseq, out.begin(), out.end(), [&](std::size_t &x) { x = static_cast<std::size_t>(round) + 1; });
  }
  std::size_t sum = 0;
  for (auto v : out)
    sum += v;
  __polyregion_offload_f1__([]() {
    __polyregion_builtin_assert(static_cast<uint32_t>(polyregion::invoke::AssertCode::Assert), "string temporary record check");
    return 0;
  });
  const std::string message = polyregion::polystl::details::lastAssert().message;
  const bool tracked = polyregion::rt_reflect::isHeap(polyregion::rt_reflect::_rt_reflect_p(message.data()).type);
  std::printf("%zu %zu %d", sum, total, tracked ? 1 : 0);
  return 0;
}
