#pragma region case: namespace-constant-policy
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#include <cstdio>

#include "test_utils.h"

namespace vendor {
struct device_policy {};
static const device_policy device;

static int consume(device_policy) { return 1; }
} // namespace vendor

int main() {
  const int result = __polyregion_offload_f1__([=]() { return vendor::consume(vendor::device); });
  std::printf("%d", result);
  return 0;
}
