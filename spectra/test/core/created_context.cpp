#pragma region case: created_context
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 14 1998 299995

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "polyregion/offload.hpp"

namespace off = polyregion::offload;

namespace {
off::Context::Device defaultDevice() {
  const auto *current = polyregion::polyrt::currentContext();
  for (const auto &device : off::Context::devices())
    if (current && current->platform->name() == device.platformName && current->device->name() == device.name) return device;
  std::fprintf(stderr, "no catalogue device matches the default context\n");
  std::exit(3);
}
} // namespace

int main() {
  const off::Context context(defaultDevice());
  std::vector<int> xs(1000, 0);
  int *p = xs.data();
  off::for_each(context, 1000, [p](std::int64_t i) { p[i] = static_cast<int>(2 * i); });
  std::vector<int> weights{0, 1, 2, 3, 4, 5, 6};
  const int *w = weights.data();
  const int sum = off::transform_reduce_unordered(
      context, 100000, 0, [w](std::int64_t i) { return w[i % 7]; }, [](int a, int b) { return a + b; });
  std::printf("%d %d %d", p[7], p[999], sum);
  return 0;
}
