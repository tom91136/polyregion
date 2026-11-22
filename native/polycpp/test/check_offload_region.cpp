#pragma region case: offload-region-fallback
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= polycpp {polycpp_defaults} {polycpp_stdpar} -DDEFAULT -o {output} {input}
#pragma region stderr-count: 0 bound to polyregion.core
#pragma region do: {output}
#pragma region requires: 117 4950 430 111 42

#pragma region case: offload-region-package
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DDEFAULT -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 5 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 117 4950 430 111 42

#pragma region case: offload-region-context-package
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DEXPLICIT -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 4 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 4950 430 21 9

#pragma region case: offload-region-context-fallback-host
#pragma region offload-only
#pragma region targets: host
#pragma region do: POLYFRONT_LIBRARY_PATH= polycpp {polycpp_defaults} {polycpp_stdpar} -DEXPLICIT -DREDUCE_ONLY -o {output} {input}
#pragma region stderr-count: 0 bound to polyregion.core
#pragma region do: {output}
#pragma region requires: 4950 430

#pragma region case: offload-region-context-fallback-refused
#pragma region offload-only
#pragma region targets: cuda opencl1_1 spirv_glcompute
#pragma region do: POLYFRONT_LIBRARY_PATH= polycpp {polycpp_defaults} {polycpp_stdpar} -DREFUSE -o {output} {input}
#pragma region do: POLYRT_HOST_FALLBACK=1 {output}
#pragma region fails-with: no kernel image matches device

#pragma region case: offload-region-context-missing-image-refused
#pragma region offload-only
#pragma region targets: cuda opencl1_1 spirv_glcompute
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-arch=host@native -DREFUSE -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 1 bound to polyregion.core
#pragma region do: POLYRT_HOST_FALLBACK=1 {output}
#pragma region fails-with: no kernel image matches device

#pragma region case: offload-region-ordered-fallback
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= polycpp {polycpp_defaults} {polycpp_stdpar} -DORDERED -o {output} {input}
#pragma region do: {output}
#pragma region requires: 99999 7 99999

#pragma region case: offload-region-extent-overflow
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= polycpp {polycpp_defaults} {polycpp_stdpar} -DEXTENT_OVERFLOW -o {output} {input}
#pragma region do: {output}
#pragma region requires: overflow overflow

#pragma region case: offload-region-context-require-offload-rejected
#pragma region offload-only
#pragma region compile-fails: -fstdpar-require-offload rejects
#pragma region do: POLYFRONT_LIBRARY_PATH= polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-require-offload -DEXPLICIT -o {output} {input}

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "polyregion/offload.hpp"

namespace off = polyregion::offload;

namespace {
int plus(int a, int b) { return a + b; }
} // namespace

#if defined(DEFAULT)
int main() {
  std::vector<int> xs(24, 0);
  int *p = xs.data();
  off::for_each(8, [p](std::int64_t i) { p[i] = static_cast<int>(2 * i); });
  const int sum =
      off::transform_reduce_unordered(100, 0, [](std::int64_t i) { return static_cast<int>(i); }, [](int a, int b) { return plus(a, b); });
  off::for_each(6, 4, [p](std::int64_t i, std::int64_t j) { p[i * 4 + j] += static_cast<int>(100 * i + j); });
  const int sum2d = off::transform_reduce_unordered(
      5, 4, 0, [](std::int64_t i, std::int64_t j) { return static_cast<int>(10 * i + j); }, [](int a, int b) { return plus(a, b); });
  off::task([p] { p[0] = 42; });
  std::printf("%d %d %d %d %d", p[7], sum, sum2d, p[5], p[0]);
  return 0;
}
#elif defined(ORDERED)
int main() {
  const int last = off::transform_reduce(100000, 0, [](std::int64_t i) { return static_cast<int>(i); }, [](int, int b) { return b; });
  const int init = off::transform_reduce(100000, 7, [](std::int64_t i) { return static_cast<int>(i); }, [](int a, int) { return a; });
  const int grid = off::transform_reduce(
      250, 400, 0, [](std::int64_t i, std::int64_t j) { return static_cast<int>(400 * i + j); }, [](int, int b) { return b; });
  std::printf("%d %d %d", last, init, grid);
  return 0;
}
#elif defined(EXTENT_OVERFLOW)
int main() {
  const std::int64_t big = std::int64_t{1} << 32;
  try {
    off::for_each(big, big, [](std::int64_t, std::int64_t) {});
    std::printf("ran");
  } catch (const std::overflow_error &) {
    std::printf("overflow");
  }
  try {
    off::for_each(big, big, 2, [](std::int64_t, std::int64_t, std::int64_t) {});
    std::printf(" ran");
  } catch (const std::overflow_error &) {
    std::printf(" overflow");
  }
  return 0;
}
#elif defined(REFUSE)
int main() {
  for (const auto &device : off::Context::devices()) {
    PolyrtDeviceInfo info{};
    if (!polyrt_device_info(device.platform, device.device, &info) || info.deviceKind != 1) continue;
    const off::Context context(device);
    const int sum = off::transform_reduce_unordered(
        context, 4, 0, [](std::int64_t i) { return static_cast<int>(i); }, [](int a, int b) { return plus(a, b); });
    std::printf("ran %d", sum);
    return 0;
  }
  std::printf("no gpu");
  return 0;
}
#elif defined(EXPLICIT)
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
  const int sum = off::transform_reduce_unordered(
      context, 100, 0, [](std::int64_t i) { return static_cast<int>(i); }, [](int a, int b) { return plus(a, b); });
  const int sum2d = off::transform_reduce_unordered(
      context, 5, 4, 0, [](std::int64_t i, std::int64_t j) { return static_cast<int>(10 * i + j); },
      [](int a, int b) { return plus(a, b); });
  #if defined(REDUCE_ONLY)
  std::printf("%d %d", sum, sum2d);
  return 0;
  #else
  uintptr_t device = 0;
  if (!polyrt_remote_malloc(context.handle(), 8 * sizeof(int), &device)) return 4;
  int *p = reinterpret_cast<int *>(device);
  off::for_each(context, 8, [p](std::int64_t i) { p[i] = static_cast<int>(3 * i); });
  off::task(context, [p] { p[0] = 9; });
  int host[8] = {};
  if (!polyrt_remote_memcpy(context.handle(), reinterpret_cast<uintptr_t>(host), device, sizeof(host), 1)) return 5;
  polyrt_remote_free(context.handle(), device);
  std::printf("%d %d %d %d", sum, sum2d, host[7], host[0]);
  return 0;
  #endif
}
#endif
