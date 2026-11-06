#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/runtime.h"

#include "kernels/generated_cpu_shared_fma.hpp"
#include "kernels/generated_vulkan_fma.hpp"
#include "test_utils.h"

using namespace polyregion::invoke;
using namespace polyregion::test_utils;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

std::vector<uint8_t> pattern(const size_t size, const uint8_t seed) {
  std::vector<uint8_t> bytes(size);
  std::iota(bytes.begin(), bytes.end(), seed);
  return bytes;
}

void write(DeviceQueue &queue, const uintptr_t handle, const size_t offset, const std::vector<uint8_t> &bytes) {
  queue.enqueueHostToDeviceAsync(bytes.data(), handle, offset, bytes.size(), {});
  queue.enqueueWaitBlocking();
}

std::vector<uint8_t> read(DeviceQueue &queue, const uintptr_t handle, const size_t offset, const size_t size) {
  std::vector<uint8_t> bytes(size);
  queue.enqueueDeviceToHostAsync(handle, offset, bytes.data(), size, {});
  queue.enqueueWaitBlocking();
  return bytes;
}

void runGrowable(Context &ctx, Backend backend, Platform &, Device &device, const ImageGroup &) {
  const auto limits = device.heapLimits();
  if (backend == Backend::Vulkan) {
    POLYTEST_REQUIRE(ctx, limits.has_value());
    POLYTEST_CHECK_S(ctx, limits->alignment >= 16 && (limits->alignment & (limits->alignment - 1)) == 0,
                     "heap alignment {} is not a power of two of at least 16", limits->alignment);
    POLYTEST_CHECK_S(ctx, storageAlignBytes(device.features()) == limits->alignment,
                     "storage_align:{} does not match the heap alignment {}", storageAlignBytes(device.features()), limits->alignment);
  }
  const size_t maxBytes = limits ? std::min<size_t>(limits->maxBytes, size_t{1} << 32) : size_t{1} << 30;
  constexpr size_t initial = 4096, grown = 3 << 20, final = (3 << 20) + 12345;

  auto queue = device.createQueue(std::chrono::seconds(30));
  const auto handle = device.reserveDevice(maxBytes, initial);
  POLYTEST_REQUIRE(ctx, handle != 0);

  const auto first = pattern(initial, 1);
  write(*queue, handle, 0, first);
  POLYTEST_CHECK(ctx, read(*queue, handle, 0, initial) == first);

  device.growDevice(handle, grown);
  POLYTEST_CHECK_S(ctx, read(*queue, handle, 0, initial) == first, "growth to {} bytes lost the committed bytes", grown);
  const auto tail = pattern(4096, 77);
  write(*queue, handle, grown - tail.size(), tail);
  POLYTEST_CHECK(ctx, read(*queue, handle, grown - tail.size(), tail.size()) == tail);

  device.growDevice(handle, final);
  POLYTEST_CHECK(ctx, read(*queue, handle, 0, initial) == first);
  POLYTEST_CHECK(ctx, read(*queue, handle, grown - tail.size(), tail.size()) == tail);

  device.growDevice(handle, initial);
  POLYTEST_CHECK_S(ctx, read(*queue, handle, grown - tail.size(), tail.size()) == tail, "a smaller target changed the committed bytes");

  try {
    device.growDevice(handle, maxBytes + 1);
    POLYTEST_FAIL(ctx, "growth past the reservation succeeded");
  } catch (const std::length_error &error) {
    POLYTEST_CHECK_S(ctx, std::string(error.what()).find(std::to_string(maxBytes)) != std::string::npos,
                     "the error `{}` does not name the {} byte limit", error.what(), maxBytes);
  }
  POLYTEST_CHECK_S(ctx, read(*queue, handle, 0, initial) == first, "a failed growth changed the committed bytes");

  if (limits) {
    try {
      (void)device.reserveDevice(limits->maxBytes + 1, initial);
      POLYTEST_FAIL(ctx, "a reservation past the device limit succeeded");
    } catch (const std::length_error &error) {
      POLYTEST_CHECK_S(ctx, std::string(error.what()).find(std::to_string(limits->maxBytes)) != std::string::npos,
                       "the error `{}` does not name the {} byte limit", error.what(), limits->maxBytes);
    }
  }

  queue.reset();
  device.freeDevice(handle);
  try {
    device.growDevice(handle, grown);
    POLYTEST_FAIL(ctx, "growth of a freed handle succeeded");
  } catch (const std::invalid_argument &) {
  }
}

std::vector<Task> discoverAll() {
  return discoverMatrix({
#ifndef __APPLE__
      {"growable-vulkan", generated::vulkan::fma, {Backend::Vulkan}, &runGrowable},
#endif
      {"growable-cpu-shared", generated::cpu_shared::fma, {Backend::SharedObject}, &runGrowable},
  });
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
