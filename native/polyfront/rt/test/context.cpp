#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "polyrt/rt.h"

#include "kernels/generated_cpu_reloc_fma.hpp"
#include "kernels/generated_cuda_fma.hpp"

using namespace polyregion::invoke;

namespace {

constexpr int32_t RelocatableBackend = static_cast<int32_t>(Backend::RelocatableObject);
constexpr int32_t CudaBackend = static_cast<int32_t>(Backend::CUDA);

struct Selection {
  int32_t platform = -1;
  int32_t device = -1;
  PolyrtDeviceInfo info{};
  explicit operator bool() const { return platform >= 0; }
};

Selection find(const int32_t backend) {
  int32_t platforms = 0;
  REQUIRE(polyrt_platform_count(&platforms));
  for (int32_t platform = 0; platform < platforms; ++platform) {
    PolyrtPlatformInfo platformInfo{};
    REQUIRE(polyrt_platform_info(platform, &platformInfo));
    if (platformInfo.backend != backend || platformInfo.deviceCount < 1) continue;
    Selection selection{.platform = platform, .device = 0};
    REQUIRE(polyrt_device_info(platform, 0, &selection.info));
    return selection;
  }
  return {};
}

struct Created {
  void *context = nullptr;
  explicit Created(const Selection &selection) { REQUIRE(polyrt_context_create(selection.platform, selection.device, &context)); }
  Created(const Created &) = delete;
  Created &operator=(const Created &) = delete;
  ~Created() {
    if (context) polyrt_context_destroy(context);
  }
  void destroy() {
    REQUIRE(polyrt_context_destroy(context));
    context = nullptr;
  }
};

const std::vector<uint8_t> &hostImage() {
  const auto &images = generated::cpu_reloc::fma;
  const auto baseline = images.find("x86-64");
  return (baseline != images.end() ? baseline : images.begin())->second;
}

const std::vector<uint8_t> &cudaImage(const PolyrtDeviceInfo &info) {
  const auto key = "sm_" + std::to_string(info.cudaArchitectureMajor) + std::to_string(info.cudaArchitectureMinor);
  const auto found = generated::cuda::fma.find(key);
  REQUIRE(found != generated::cuda::fma.end());
  return found->second;
}

std::optional<float> tryFma(void *context, const Selection &selection, const std::vector<uint8_t> &image, const char *module, const float a,
                            const float b, const float c) {
  if (!polyrt_context_acquire(context)) return std::nullopt;
  std::optional<float> result;
  uintptr_t out = 0;
  float x = a, y = b, z = c;
  const std::array<uint8_t, 4> types{static_cast<uint8_t>(Type::Float32), static_cast<uint8_t>(Type::Float32),
                                     static_cast<uint8_t>(Type::Float32), static_cast<uint8_t>(Type::Ptr)};
  if (polyrt_remote_load(context, module, selection.info.platformKind, selection.info.moduleFormat, 0, nullptr, image.size(), image.data())
      && polyrt_remote_malloc(context, sizeof(float), &out)) {
    const std::array<void *, 4> arguments{&x, &y, &z, &out};
    float value = 0;
    if (polyrt_remote_launch(context, module, "_fma", 1, 1, 1, 0, 0, 0, 0, types.size(), types.data(), arguments.data())
        && polyrt_remote_memcpy(context, reinterpret_cast<uintptr_t>(&value), out, sizeof(value), 1))
      result = value;
    polyrt_remote_free(context, out);
  }
  if (!polyrt_context_release(context)) return std::nullopt;
  return result;
}

float launchFma(void *context, const Selection &selection, const std::vector<uint8_t> &image, const char *module, const float a,
                const float b, const float c) {
  const auto result = tryFma(context, selection, image, module, a, b, c);
  INFO(polyrt_error_message());
  REQUIRE(result.has_value());
  return *result;
}

const std::vector<uint8_t> &imageFor(const Selection &selection) {
  return selection.info.backend == CudaBackend ? cudaImage(selection.info) : hostImage();
}

Selection defaultSelection() {
  auto *context = polyregion::polyrt::currentContext();
  REQUIRE(context);
  Selection selection{.platform = 0, .device = 0};
  selection.info.backend = context->platform->kind() == polyregion::polyrt::PlatformKind::HostThreaded ? RelocatableBackend : CudaBackend;
  selection.info.platformKind = static_cast<int32_t>(context->platform->kind());
  selection.info.moduleFormat = static_cast<int32_t>(context->device->moduleFormat());
  selection.info.cudaArchitectureMajor = static_cast<int32_t>(context->device->cudaArchitectureMajor());
  selection.info.cudaArchitectureMinor = static_cast<int32_t>(context->device->cudaArchitectureMinor());
  return selection;
}

void loadConcurrently(const Selection &selection) {
  Created first(selection), second(selection);
  const auto &image = imageFor(selection);
  std::barrier sync(2);
  std::atomic<int> failures{0};
  const auto run = [&](void *context, const float offset) {
    for (int k = 0; k < 64; ++k) {
      const auto name = "concurrent_" + std::to_string(k);
      sync.arrive_and_wait();
      const auto result = tryFma(context, selection, image, name.c_str(), 1.f, offset, 1.f);
      if (!result || *result != 1.f + offset) ++failures;
    }
  };
  std::thread a(run, first.context, 1.f), b(run, second.context, 2.f);
  a.join();
  b.join();
  CHECK(failures == 0);
  CHECK(launchFma(first.context, selection, image, "concurrent_0", 2.f, 2.f, 2.f) == 6.f);
  CHECK(launchFma(second.context, selection, image, "concurrent_0", 3.f, 2.f, 2.f) == 8.f);
}

std::string lastError() { return polyrt_error_message(); }

} // namespace

TEST_CASE("enumeration lists the host platform and its devices") {
  int32_t platforms = 0;
  REQUIRE(polyrt_platform_count(&platforms));
  REQUIRE(platforms >= 1);
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  CHECK(host.info.deviceKind == static_cast<int32_t>(polyregion::polyrt::DeviceKind::CPU));
  CHECK(host.info.backend == RelocatableBackend);
  CHECK(std::strlen(host.info.name) > 0);
  PolyrtPlatformInfo platformInfo{};
  REQUIRE(polyrt_platform_info(host.platform, &platformInfo));
  CHECK(std::strlen(platformInfo.name) > 0);
  CHECK(platformInfo.deviceCount >= 1);
}

TEST_CASE("enumeration is stable across calls") {
  int32_t first = 0, second = 0;
  REQUIRE(polyrt_platform_count(&first));
  REQUIRE(polyrt_platform_count(&second));
  CHECK(first == second);
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  PolyrtDeviceInfo again{};
  REQUIRE(polyrt_device_info(host.platform, host.device, &again));
  CHECK(std::string(again.name) == host.info.name);
}

TEST_CASE("invalid indices and arguments report an error") {
  int32_t platforms = 0;
  REQUIRE(polyrt_platform_count(&platforms));
  PolyrtPlatformInfo platformInfo{};
  PolyrtDeviceInfo deviceInfo{};
  void *context = reinterpret_cast<void *>(0x1);

  CHECK_FALSE(polyrt_platform_count(nullptr));
  CHECK_FALSE(polyrt_platform_info(-1, &platformInfo));
  CHECK_FALSE(polyrt_platform_info(platforms, &platformInfo));
  CHECK_FALSE(polyrt_platform_info(0, nullptr));
  CHECK_FALSE(polyrt_device_info(-1, 0, &deviceInfo));
  CHECK_FALSE(polyrt_device_info(platforms, 0, &deviceInfo));

  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  PolyrtPlatformInfo hostInfo{};
  REQUIRE(polyrt_platform_info(host.platform, &hostInfo));
  CHECK_FALSE(polyrt_device_info(host.platform, -1, &deviceInfo));
  CHECK_FALSE(polyrt_device_info(host.platform, hostInfo.deviceCount, &deviceInfo));
  CHECK_FALSE(polyrt_device_info(host.platform, 0, nullptr));

  CHECK_FALSE(polyrt_context_create(platforms, 0, &context));
  CHECK(lastError().find("platform") != std::string::npos);
  CHECK(context == nullptr);
  context = reinterpret_cast<void *>(0x1);
  CHECK_FALSE(polyrt_context_create(host.platform, hostInfo.deviceCount, &context));
  CHECK(lastError().find("device") != std::string::npos);
  CHECK(context == nullptr);
  CHECK_FALSE(polyrt_context_create(host.platform, 0, nullptr));
}

TEST_CASE("a created host context reports its device and releases once") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  Created created(host);
  REQUIRE(created.context != nullptr);
  CHECK(polyrt_device_kind(created.context) == polyregion::polyrt::DeviceKind::CPU);
  void *released = created.context;
  created.destroy();
  CHECK_FALSE(polyrt_context_destroy(released));
  CHECK_FALSE(lastError().empty());
}

TEST_CASE("a released handle is rejected by every entry point") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  void *released = nullptr;
  {
    Created created(host);
    released = created.context;
  }
  CHECK_FALSE(polyrt_context_acquire(released));
  CHECK(lastError().find("unknown or released") != std::string::npos);
  CHECK_FALSE(polyrt_context_release(released));
  CHECK_FALSE(polyrt_context_destroy(released));
  PolyrtStats stats{};
  CHECK_FALSE(polyrt_stats(released, &stats));
  uintptr_t pointer = 0;
  CHECK_FALSE(polyrt_remote_malloc(released, 8, &pointer));
  CHECK(lastError().find("unknown or released") != std::string::npos);
  CHECK_FALSE(polyrt_context_acquire(reinterpret_cast<void *>(0x1000)));
}

TEST_CASE("handles are never reused") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  std::vector<void *> seen;
  for (int i = 0; i < 8; ++i) {
    Created created(host);
    for (auto *earlier : seen)
      CHECK(earlier != created.context);
    seen.emplace_back(created.context);
  }
  Created live(host);
  for (auto *earlier : seen) {
    CHECK_FALSE(polyrt_context_destroy(earlier));
    CHECK_FALSE(polyrt_context_acquire(earlier));
  }
  PolyrtStats stats{};
  CHECK(polyrt_stats(live.context, &stats));
}

TEST_CASE("a context held by a transaction on this thread cannot be destroyed") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  Created created(host);
  REQUIRE(polyrt_context_acquire(created.context));
  CHECK_FALSE(polyrt_context_destroy(created.context));
  CHECK(lastError().find("transaction") != std::string::npos);
  REQUIRE(polyrt_context_release(created.context));
  created.destroy();
}

TEST_CASE("only the acquiring thread can release a transaction") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  Created created(host);
  REQUIRE(polyrt_context_acquire(created.context));
  bool otherReleased = true;
  std::string otherMessage;
  std::thread([&] {
    otherReleased = polyrt_context_release(created.context);
    otherMessage = lastError();
  }).join();
  CHECK_FALSE(otherReleased);
  CHECK(otherMessage.find("no matching acquire") != std::string::npos);
  REQUIRE(polyrt_context_release(created.context));
  CHECK_FALSE(polyrt_context_release(created.context));
}

TEST_CASE("release reports a mirror refusal recorded during the transaction") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  Created created(host);
  REQUIRE(polyrt_context_acquire(created.context));
  REQUIRE(polyregion::polyrt::sma::foreignContextActive("probe"));
  CHECK_FALSE(polyrt_context_release(created.context));
  CHECK(lastError().find("created context") != std::string::npos);
  REQUIRE(polyrt_context_acquire(created.context));
  CHECK(polyrt_context_release(created.context));
}

TEST_CASE("destroying a context that was not created reports an error") {
  CHECK_FALSE(polyrt_context_destroy(nullptr));
  CHECK_FALSE(lastError().empty());
  auto *current = polyrt_context_current();
  if (current) {
    CHECK_FALSE(polyrt_context_destroy(current));
    CHECK_FALSE(lastError().empty());
  }
}

TEST_CASE("a created host context launches a kernel") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  Created created(host);
  CHECK(launchFma(created.context, host, hostImage(), "fma_created", 2.f, 3.f, 4.f) == 10.f);
  PolyrtStats stats{};
  REQUIRE(polyrt_stats(created.context, &stats));
  CHECK(stats.launches == 1);
}

TEST_CASE("contexts count their own launches") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  Created first(host), second(host);
  REQUIRE(first.context != second.context);
  CHECK(launchFma(first.context, host, hostImage(), "fma_first", 1.f, 2.f, 3.f) == 5.f);
  CHECK(launchFma(first.context, host, hostImage(), "fma_first", 2.f, 2.f, 3.f) == 7.f);
  CHECK(launchFma(second.context, host, hostImage(), "fma_second", 3.f, 2.f, 3.f) == 9.f);
  PolyrtStats a{}, b{};
  REQUIRE(polyrt_stats(first.context, &a));
  REQUIRE(polyrt_stats(second.context, &b));
  CHECK(a.launches == 2);
  CHECK(b.launches == 1);
}

TEST_CASE("releasing a created context leaves the default context usable") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  const auto fallback = defaultSelection();
  auto *current = polyrt_context_current();
  REQUIRE(current);
  const auto &defaultImage = imageFor(fallback);
  CHECK(launchFma(current, fallback, defaultImage, "fma_default_before", 1.f, 2.f, 3.f) == 5.f);
  PolyrtStats before{};
  REQUIRE(polyrt_stats(current, &before));
  {
    Created created(host);
    CHECK(launchFma(created.context, host, hostImage(), "fma_beside_default", 1.f, 1.f, 1.f) == 2.f);
    CHECK(polyrt_context_current() == current);
    created.destroy();
  }
  CHECK(polyrt_context_current() == current);
  CHECK(launchFma(current, fallback, defaultImage, "fma_default_after", 2.f, 2.f, 3.f) == 7.f);
  PolyrtStats after{};
  REQUIRE(polyrt_stats(current, &after));
  CHECK(after.launches == before.launches + 1);
}

TEST_CASE("destroying a context with live allocations succeeds") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  Created created(host);
  uintptr_t pointer = 0;
  REQUIRE(polyrt_remote_malloc(created.context, 64, &pointer));
  REQUIRE(pointer != 0);
  created.destroy();
}

TEST_CASE("a created context loads modules onto its own device") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  Created created(host);
  const auto &image = hostImage();
  REQUIRE(polyrt_context_acquire(created.context));
  CHECK(polyrt_remote_load(created.context, "fma_loaded", host.info.platformKind, host.info.moduleFormat, 0, nullptr, image.size(),
                           image.data()));
  CHECK_FALSE(polyrt_remote_load(created.context, "fma_mismatch", host.info.platformKind, host.info.moduleFormat + 1, 0, nullptr,
                                 image.size(), image.data()));
  REQUIRE(polyrt_context_release(created.context));
}

TEST_CASE("a created CUDA context launches a kernel") {
  const auto cuda = find(CudaBackend);
  if (!cuda) SKIP("no CUDA device");
  Created created(cuda);
  CHECK(launchFma(created.context, cuda, cudaImage(cuda.info), "fma_cuda", 2.f, 3.f, 4.f) == 10.f);
  CHECK(polyrt_device_kind(created.context) == polyregion::polyrt::DeviceKind::GPU);
}

TEST_CASE("a created CUDA context is released while the default context is live") {
  const auto cuda = find(CudaBackend);
  if (!cuda) SKIP("no CUDA device");
  auto *current = polyrt_context_current();
  const auto fallback = defaultSelection();
  CHECK(launchFma(current, fallback, imageFor(fallback), "fma_cuda_default_before", 1.f, 1.f, 1.f) == 2.f);
  {
    Created created(cuda);
    CHECK(launchFma(created.context, cuda, cudaImage(cuda.info), "fma_cuda_beside_default", 1.f, 5.f, 1.f) == 6.f);
    created.destroy();
  }
  CHECK(polyrt_context_current() == current);
  CHECK(launchFma(current, fallback, imageFor(fallback), "fma_cuda_default_after", 1.f, 2.f, 1.f) == 3.f);
  Created again(cuda);
  CHECK(launchFma(again.context, cuda, cudaImage(cuda.info), "fma_cuda_again", 2.f, 5.f, 1.f) == 11.f);
}

TEST_CASE("destroying a CUDA context returns its device memory") {
  const auto cuda = find(CudaBackend);
  if (!cuda) SKIP("no CUDA device");
  const auto chunk = static_cast<size_t>(cuda.info.globalMemoryBytes / 4);
  for (int i = 0; i < 16; ++i) {
    Created created(cuda);
    uintptr_t pointer = 0;
    REQUIRE(polyrt_remote_malloc(created.context, chunk, &pointer));
    REQUIRE(pointer != 0);
    created.destroy();
  }
}

TEST_CASE("two contexts load the same modules concurrently on a host device") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  loadConcurrently(host);
}

TEST_CASE("two contexts load the same modules concurrently on a CUDA device") {
  const auto cuda = find(CudaBackend);
  if (!cuda) SKIP("no CUDA device");
  loadConcurrently(cuda);
}

TEST_CASE("a module store tolerates concurrent loads and lookups") {
  polyregion::invoke::detail::ModuleStore<int, int> store(
      "test", [](const std::string &image) { return static_cast<int>(image.size()); },
      [](const int module, const std::string &symbol, const std::vector<Type> &) { return module + static_cast<int>(symbol.size()); });
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t)
    threads.emplace_back([&, t] {
      for (int k = 0; k < 2000; ++k) {
        const auto name = "m" + std::to_string(t) + "_" + std::to_string(k);
        store.loadModule(name, std::string(static_cast<size_t>(k % 7 + 1), 'x'));
        if (!store.moduleLoaded(name)) ++failures;
        if (store.resolveFunction(name, "sym", {}) != k % 7 + 1 + 3) ++failures;
        if (!store.find(name)) ++failures;
      }
    });
  for (auto &thread : threads)
    thread.join();
  CHECK(failures == 0);
}

TEST_CASE("contexts on separate threads launch concurrently") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i)
    threads.emplace_back([&, i] {
      void *context = nullptr;
      if (!polyrt_context_create(host.platform, host.device, &context)) {
        ++failures;
        return;
      }
      const auto name = "threaded_" + std::to_string(i);
      for (int k = 0; k < 8; ++k) {
        const auto result = tryFma(context, host, hostImage(), name.c_str(), 2.f, static_cast<float>(k), 1.f);
        if (!result || *result != 2.f * static_cast<float>(k) + 1.f) ++failures;
      }
      if (!polyrt_context_destroy(context)) ++failures;
    });
  for (auto &thread : threads)
    thread.join();
  CHECK(failures == 0);
}

TEST_CASE("creating and destroying contexts concurrently succeeds") {
  const auto host = find(RelocatableBackend);
  REQUIRE(host);
  std::vector<std::thread> threads;
  std::array<bool, 4> ok{};
  for (size_t i = 0; i < ok.size(); ++i)
    threads.emplace_back([&, i] {
      void *context = nullptr;
      bool good = polyrt_context_create(host.platform, host.device, &context);
      good = good && polyrt_context_destroy(context);
      ok[i] = good;
    });
  for (auto &thread : threads)
    thread.join();
  for (const auto good : ok)
    CHECK(good);
}
