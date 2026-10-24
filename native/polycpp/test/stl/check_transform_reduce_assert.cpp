#pragma region case: transform-reduce-assert-clean
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 5050

#pragma region case: transform-reduce-assert-raised
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DRAISE -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#include <cstdint>
#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

#include "polyregion/enums.h"
#include "polystl/polystl.h"

int checked(int x) {
  if (x < 0) __polyregion_builtin_assert(static_cast<uint32_t>(polyregion::invoke::AssertCode::Assert), "reduce");
  return x;
}

int main() {
  std::vector<int> xs(100);
  std::iota(xs.begin(), xs.end(), 1);
#ifdef RAISE
  xs[6] = -1;
#endif
  const int sum =
      std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return checked(x); });
  const bool raised = polyregion::polystl::details::lastAssert().raised;
#ifdef RAISE
  std::printf("%d", raised ? 1 : 0);
  (void)sum;
#else
  std::printf("%d %d", raised ? 1 : 0, sum);
#endif
  return 0;
}
