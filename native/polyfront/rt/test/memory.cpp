#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#ifndef _WIN32
  #include <unistd.h>
#endif

#include "fmt/args.h"
#include <catch2/catch_test_macros.hpp>

#ifdef POLYRT_TESTS_POLYDCO
  #include "polydco/polydco.h"
#endif
#include "polyinvoke/object_platform.h"
#include "polyregion/conventions.h"
#include "polyregion/show.hpp"
#include "polyrt/assert_buffer.hpp"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"
#include "polyrt/sma_runtime.hpp"

#include "jit_policy.hpp"
#include "reflect-rt/rt_reflect.hpp"

using namespace polyregion::invoke;
using namespace polyregion::runtime;

namespace {

struct StubDevice final : object::ObjectDevice {
  bool shared;
  bool cpu;
  size_t threads;
  size_t subgroup;
  size_t localMemory = 12345;
  size_t globalMemory = 67890;
  size_t units = 7;
  uint32_t cudaMajor = 9;
  uint32_t cudaMinor = 1;
  ModuleFormat format;
  std::optional<std::vector<std::string>> advertised;
  bool failRemoteFree = false;
  bool failRemoteMalloc = false;
  bool reuseRemoteAddress = false;
  uintptr_t reusableRemoteAddress = 0x1000;
  bool packAllocations = false;
  std::optional<HeapLimits> heap;
  std::optional<size_t> viewStart;
  std::vector<std::max_align_t> pool = std::vector<std::max_align_t>(256);
  size_t poolUsed = 0;
  size_t sharedAllocs = 0, sharedFrees = 0, remoteAllocs = 0, remoteFrees = 0;

  void *carve(const size_t size) {
    constexpr size_t quantum = 16;
    poolUsed = (poolUsed + quantum - 1) / quantum * quantum;
    void *block = reinterpret_cast<char *>(pool.data()) + poolUsed;
    poolUsed += size;
    return block;
  }

  explicit StubDevice(const bool shared, const bool cpu = true, const size_t threads = 1024, const size_t subgroup = 1,
                      const ModuleFormat format = ModuleFormat::Object)
      : shared(shared), cpu(cpu), threads(threads), subgroup(subgroup), format(format) {}
  std::string name() override { return "stub"; }
  std::vector<std::string> features() override { return advertised ? *advertised : ObjectDevice::features(); }
  PhysicalDevice physicalDevice() override { return cpu ? PhysicalDevice::host() : PhysicalDevice::synthetic(Backend::Metal, 0); }
  bool sharedAddressSpace() override { return shared; }
  ModuleFormat moduleFormat() override { return format; }
  size_t maxThreadsPerBlock() override { return threads; }
  size_t subgroupSize() override { return subgroup; }
  size_t localMemoryBytes() override { return localMemory; }
  size_t globalMemoryBytes() override { return globalMemory; }
  size_t computeUnits() override { return units; }
  uint32_t cudaArchitectureMajor() override { return cudaMajor; }
  uint32_t cudaArchitectureMinor() override { return cudaMinor; }
  void loadModule(const std::string &, const std::string &) override {}
  bool moduleLoaded(const std::string &) override { return false; }
  std::optional<void *> mallocShared(size_t size, Access access) override {
    if (!shared) return {};
    sharedAllocs++;
    if (packAllocations) return carve(size);
    return ObjectDevice::mallocShared(size, access);
  }
  void freeShared(void *ptr) override {
    sharedFrees++;
    if (!packAllocations) ObjectDevice::freeShared(ptr);
  }
  uintptr_t mallocDevice(size_t size, Access access) override {
    remoteAllocs++;
    if (failRemoteMalloc) throw std::runtime_error("injected allocation failure");
    if (reuseRemoteAddress) return reusableRemoteAddress;
    if (packAllocations) return reinterpret_cast<uintptr_t>(carve(size));
    return ObjectDevice::mallocDevice(size, access);
  }
  void freeDevice(uintptr_t ptr) override {
    remoteFrees++;
    if (failRemoteFree) throw std::runtime_error("injected free failure");
    if (reuseRemoteAddress || packAllocations) return;
    ObjectDevice::freeDevice(ptr);
  }
  std::optional<HeapLimits> heapLimits() override { return heap; }
  std::optional<size_t> arenaViewStart(const std::string &) override { return viewStart; }
  std::unique_ptr<DeviceQueue> createQueue(const std::chrono::duration<int64_t> &) override { return {}; }
};

struct StubQueue final : object::ObjectDeviceQueue {
  using ObjectDeviceQueue::ObjectDeviceQueue;

  bool failInvoke = false;
  bool mutateFirstPointer = false;
  bool raiseAssert = false;
  std::vector<Type> invokedTypes;
  std::vector<std::byte> invokedData;
  uintptr_t firstPointer = 0;
  std::string operations;
  bool idle = false;
  std::function<void()> onInvoke;
  std::function<void()> onCopy;

  void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) override {
    operations += 'H';
    ObjectDeviceQueue::enqueueHostToDeviceAsync(src, dst, dstOffset, size, cb);
  }
  void enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                  const MaybeCallback &cb) override {
    if (onCopy) onCopy();
    ObjectDeviceQueue::enqueueDeviceToDeviceAsync(src, srcOffset, dst, dstOffset, size, cb);
  }
  void enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t bytes, const MaybeCallback &cb) override {
    operations += 'D';
    ObjectDeviceQueue::enqueueDeviceToHostAsync(src, srcOffset, dst, bytes, cb);
  }
  void enqueueWaitBlocking() override {
    operations += 'W';
    ObjectDeviceQueue::enqueueWaitBlocking();
  }
  bool quiescent() override { return idle; }

  void enqueueInvokeAsync(const std::string &, const std::string &, const std::vector<Type> &types, std::vector<std::byte> data,
                          const Policy &, const MaybeCallback &) override {
    operations += 'I';
    if (failInvoke) throw std::runtime_error("injected launch failure");
    invokedTypes = types;
    invokedData = data;
    if (!types.empty() && types.front() == Type::Ptr) {
      const auto arguments = detail::argDataAsPointers(types, data);
      std::memcpy(&firstPointer, arguments.front(), sizeof(firstPointer));
    }
    if (raiseAssert) {
      const auto arguments = detail::argDataAsPointers(types, data);
      uintptr_t error = 0;
      std::memcpy(&error, arguments.front(), sizeof(error));
      const uint8_t reported[] = {1, 0, 0, 0, 'b', 'o', 'o', 'm', 0};
      std::memcpy(reinterpret_cast<void *>(error), reported, sizeof(reported));
    }
    if (mutateFirstPointer) {
      const auto arguments = detail::argDataAsPointers(types, data);
      const auto pointerArgument = std::ranges::find(types, Type::Ptr);
      REQUIRE(pointerArgument != types.end());
      uintptr_t pointer = 0;
      std::memcpy(&pointer, arguments[std::distance(types.begin(), pointerArgument)], sizeof(pointer));
      *reinterpret_cast<uint64_t *>(pointer) = 0x123456789ABCDEF0ull;
    }
    if (onInvoke) onInvoke();
  }
};

struct WithStubDevice {
  std::unique_ptr<Device> previous;
  StubDevice *stub;

  explicit WithStubDevice(const bool shared, const bool cpu = true, const size_t threads = 1024, const size_t subgroup = 1,
                          const ModuleFormat format = ModuleFormat::Object) {
    polyregion::polyrt::initialise();
    previous = std::move(polyregion::polyrt::currentDevice);
    auto owned = std::make_unique<StubDevice>(shared, cpu, threads, subgroup, format);
    stub = owned.get();
    polyregion::polyrt::currentDevice = std::move(owned);
  }
  ~WithStubDevice() { polyregion::polyrt::currentDevice = std::move(previous); }
};

struct StubManagedPlatform final : Platform {
  std::string name() override { return "stub"; }
  std::vector<Property> properties() override { return {}; }
  PlatformKind kind() override { return PlatformKind::Managed; }
  std::vector<std::unique_ptr<Device>> enumerate() override { return {}; }
};

std::unique_ptr<Platform> takeCurrentPlatform() {
  polyregion::polyrt::initialise();
  return std::move(polyregion::polyrt::currentPlatform);
}

struct WithStubManagedPlatform {
  std::unique_ptr<Platform> previousPlatform = takeCurrentPlatform();
  std::unique_ptr<DeviceQueue> previousQueue = std::move(polyregion::polyrt::currentQueue);
  WithStubManagedPlatform() {
    polyregion::polyrt::currentPlatform = std::make_unique<StubManagedPlatform>();
    polyregion::polyrt::currentQueue = std::make_unique<StubQueue>(std::chrono::seconds(1));
  }
  ~WithStubManagedPlatform() {
    polyregion::polyrt::currentQueue = std::move(previousQueue);
    polyregion::polyrt::currentPlatform = std::move(previousPlatform);
  }
};

} // namespace

#ifdef POLYRT_TESTS_POLYDCO
extern "C" int __polyregion_fc_assert_raised();
static bool assertReported() { return __polyregion_fc_assert_raised() != 0; }
#else
static std::atomic<uint32_t> reportedAssertCode{0};
extern "C" void __polyregion_builtin_assert(const uint32_t code, const char *) { reportedAssertCode = code; }
static bool assertReported() { return reportedAssertCode.exchange(0) != 0; }
#endif

static uintptr_t remoteAlloc(void *context, size_t bytes) {
  uintptr_t result = 0;
  REQUIRE(polyrt_remote_malloc(context, bytes, &result));
  return result;
}

static uintptr_t remoteTempAlloc(void *context, size_t bytes) {
  uintptr_t result = 0;
  REQUIRE(polyrt_remote_temp_malloc(context, bytes, &result));
  return result;
}

TEST_CASE("device queries describe the selected runtime device") {
  {
    WithStubDevice device(false, true, 73);
    polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
    CHECK(polyrt_device_max_threads_per_block(&context) == 73);
    CHECK(polyrt_device_subgroup_size(&context) == 1);
    CHECK(polyrt_device_local_memory_bytes(&context) == 12345);
    CHECK(polyrt_device_global_memory_bytes(&context) == 67890);
    CHECK(polyrt_device_compute_units(&context) == 7);
    CHECK(polyrt_device_cuda_architecture_major(&context) == 9);
    CHECK(polyrt_device_cuda_architecture_minor(&context) == 1);
    CHECK(polyrt_device_kind(&context) == polyregion::polyrt::DeviceKind::CPU);
  }
  {
    WithStubDevice device(false, false, 511, 64);
    polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
    CHECK(polyrt_device_max_threads_per_block(&context) == 511);
    CHECK(polyrt_device_subgroup_size(&context) == 64);
    CHECK(polyrt_device_kind(&context) == polyregion::polyrt::DeviceKind::GPU);
  }
  {
    WithStubDevice device(false, false, 511, 16, ModuleFormat::SPIRV_Kernel);
    polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
    CHECK(polyrt_device_subgroup_size(&context) == 16);
  }
  {
    WithStubDevice device(false, false, 511, 64, ModuleFormat::SPIRV_GLCompute);
    polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
    CHECK(polyrt_device_subgroup_size(&context) == 64);
  }
  {
    WithStubDevice device(false, true, 2048, 1, ModuleFormat::Source);
    polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
    CHECK(polyrt_device_subgroup_size(&context) == 1);
  }
}

TEST_CASE("failed remote frees remain tracked for retry") {
  WithStubDevice device(false);
  polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
  const auto remote = remoteAlloc(&context, 8);
  device.stub->failRemoteFree = true;
  CHECK_FALSE(polyrt_remote_free(&context, remote));
  CHECK(std::string(polyrt_error_message()).find("injected free failure") != std::string::npos);
  CHECK(context.remoteAllocations.contains(remote));
  device.stub->failRemoteFree = false;
  REQUIRE(polyrt_remote_free(&context, remote));
  CHECK(context.remoteAllocations.empty());
}

TEST_CASE("remote allocation C ABI reports failures without unwinding") {
  static_assert(noexcept(polyrt_remote_malloc(nullptr, 0, nullptr)));
  static_assert(noexcept(polyrt_remote_temp_malloc(nullptr, 0, nullptr)));
  static_assert(noexcept(polyrt_remote_free(nullptr, 0)));
  static_assert(noexcept(polyrt_remote_memcpy(nullptr, 0, 0, 0, 0)));

  WithStubDevice device(false);
  polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
  uintptr_t result = 123;

  CHECK_FALSE(polyrt_remote_temp_malloc(&context, 8, &result));
  CHECK(result == 0);
  CHECK(std::string(polyrt_error_message()).find("no active context transaction") != std::string::npos);

  device.stub->failRemoteMalloc = true;
  result = 123;
  CHECK_FALSE(polyrt_remote_malloc(&context, 8, &result));
  CHECK(result == 0);
  CHECK(std::string(polyrt_error_message()).find("injected allocation failure") != std::string::npos);
  CHECK(context.remoteAllocations.empty());
}

TEST_CASE("context transactions release only temporary allocations created within their scope") {
  WithStubDevice device(false);
  polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
  const auto persistent = remoteAlloc(&context, 8);
  REQUIRE(polyrt_context_acquire(&context));
  const auto returned = remoteAlloc(&context, 16);
  const auto outer = remoteTempAlloc(&context, 16);
  REQUIRE(polyrt_context_acquire(&context));
  const auto inner = remoteTempAlloc(&context, 32);
  REQUIRE(polyrt_context_release(&context));
  CHECK(context.remoteAllocations.contains(persistent));
  CHECK(context.remoteAllocations.contains(returned));
  CHECK(context.remoteAllocations.contains(outer));
  CHECK_FALSE(context.remoteAllocations.contains(inner));
  REQUIRE(polyrt_context_release(&context));
  CHECK(context.remoteAllocations.size() == 2);
  CHECK(context.remoteAllocations.contains(persistent));
  CHECK(context.remoteAllocations.contains(returned));
  REQUIRE(polyrt_remote_free(&context, returned));
  REQUIRE(polyrt_remote_free(&context, persistent));
  CHECK(context.remoteAllocations.empty());
}

TEST_CASE("failed hidden temporary cleanup is retried by the next transaction") {
  WithStubDevice device(false);
  polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
  REQUIRE(polyrt_context_acquire(&context));
  const auto temporary = remoteTempAlloc(&context, 8);
  device.stub->failRemoteFree = true;
  CHECK_THROWS(polyrt_context_release_or_throw(&context));
  CHECK(context.remoteAllocations.contains(temporary));
  CHECK(context.remoteAllocations.at(temporary).cleanupPending);
  CHECK_THROWS(polyrt_context_acquire_or_throw(&context));
  CHECK(context.remoteAllocations.contains(temporary));

  device.stub->failRemoteFree = false;
  REQUIRE(polyrt_context_acquire(&context));
  CHECK(context.remoteAllocations.empty());
  REQUIRE(polyrt_context_release(&context));
  CHECK(device.stub->remoteFrees == 3);
}

TEST_CASE("context C ABI reports failures without unwinding") {
  static_assert(noexcept(polyrt_context_acquire(nullptr)));
  static_assert(noexcept(polyrt_context_release(nullptr)));

  WithStubDevice device(false);
  polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};
  CHECK_FALSE(polyrt_context_release(&context));
  CHECK(std::string(polyrt_error_message()).find("no matching acquire") != std::string::npos);
  CHECK(polyrt_context_acquire(&context));
  CHECK(std::string(polyrt_error_message()).empty());
  const auto temporary = remoteTempAlloc(&context, 8);
  device.stub->failRemoteFree = true;
  CHECK_FALSE(polyrt_context_release(&context));
  CHECK(std::string(polyrt_error_message()).find("injected free failure") != std::string::npos);
  CHECK(context.remoteAllocations.at(temporary).cleanupPending);
  CHECK_FALSE(polyrt_context_acquire(&context));
  CHECK(context.remoteAllocations.contains(temporary));
  device.stub->failRemoteFree = false;
  CHECK(polyrt_context_acquire(&context));
  CHECK(context.remoteAllocations.empty());
  CHECK(polyrt_context_release(&context));
}

TEST_CASE("released temporary addresses are not retained by enclosing scopes") {
  WithStubDevice device(false);
  device.stub->reuseRemoteAddress = true;
  polyregion::polyrt::ExecutionContext context{nullptr, device.stub, nullptr};

  SECTION("nested scope cleanup") {
    REQUIRE(polyrt_context_acquire(&context));
    REQUIRE(polyrt_context_acquire(&context));
    const auto temporary = remoteTempAlloc(&context, 8);
    REQUIRE(polyrt_context_release(&context));
    const auto persistent = remoteAlloc(&context, 8);
    REQUIRE(persistent == temporary);
    REQUIRE(polyrt_context_release(&context));
    CHECK(context.remoteAllocations.contains(persistent));
    CHECK(device.stub->remoteFrees == 1);
    REQUIRE(polyrt_remote_free(&context, persistent));
  }

  SECTION("explicit cleanup") {
    REQUIRE(polyrt_context_acquire(&context));
    const auto temporary = remoteTempAlloc(&context, 8);
    REQUIRE(polyrt_remote_free(&context, temporary));
    const auto persistent = remoteAlloc(&context, 8);
    REQUIRE(persistent == temporary);
    REQUIRE(polyrt_context_release(&context));
    CHECK(context.remoteAllocations.contains(persistent));
    CHECK(device.stub->remoteFrees == 1);
    REQUIRE(polyrt_remote_free(&context, persistent));
  }

  CHECK(context.remoteAllocations.empty());
  CHECK(device.stub->remoteFrees == 2);
}

TEST_CASE("device memset and USM host access use the supplied execution context") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  polyregion::polyrt::ExecutionContext context{nullptr, device.stub, &queue};
  std::array<uint8_t, 5> remote{1, 2, 3, 4, 5};

  polyrt_device_memset(&context, remote.data(), 0xA5, remote.size());
  CHECK(remote == std::array<uint8_t, 5>{0xA5, 0xA5, 0xA5, 0xA5, 0xA5});

  auto *local = static_cast<uint8_t *>(polyrt_device_usm_host_acquire(&context, remote.data(), remote.size(), 1));
  REQUIRE(local);
  for (size_t i = 0; i < remote.size(); ++i) {
    CHECK(local[i] == remote[i]);
    local[i] = 0x3C;
  }
  polyrt_device_usm_host_release(&context, remote.data(), local, remote.size(), 2);
  CHECK(remote == std::array<uint8_t, 5>{0x3C, 0x3C, 0x3C, 0x3C, 0x3C});
}

TEST_CASE("remote launch releases mirrored arguments after a failure") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  queue.failInvoke = true;
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  std::array<uint8_t, 4> first{};
  std::array<uint8_t, 8> second{};
  const std::array<uint8_t, 2> types{static_cast<uint8_t>(Type::Ptr), static_cast<uint8_t>(Type::Ptr)};
  const std::array<void *, 2> arguments{first.data(), second.data()};
  const std::array<size_t, 2> mirrorSizes{first.size(), second.size()};
  const std::array<uint8_t, 2> mirrorKinds{1, 1};
  CHECK_FALSE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                              arguments.data(), mirrorSizes.data(), mirrorKinds.data(), nullptr));
  CHECK(device.stub->remoteAllocs == 2);
  CHECK(device.stub->remoteFrees == 2);
  CHECK(context.remoteAllocations.empty());
}

TEST_CASE("failed mirror cleanup is retried by the next transaction") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  std::array<uint8_t, 4> local{};
  const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::Ptr)};
  const std::array<void *, 1> arguments{local.data()};
  const std::array<size_t, 1> mirrorSizes{local.size()};
  const std::array<uint8_t, 1> mirrorKinds{1};
  device.stub->failRemoteFree = true;
  CHECK_FALSE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                              arguments.data(), mirrorSizes.data(), mirrorKinds.data(), nullptr));
  REQUIRE(context.remoteAllocations.size() == 1);
  CHECK(context.remoteAllocations.begin()->second.cleanupPending);

  device.stub->failRemoteFree = false;
  REQUIRE(polyrt_context_acquire(&context));
  CHECK(context.remoteAllocations.empty());
  REQUIRE(polyrt_context_release(&context));
  CHECK(device.stub->remoteFrees == 2);
}

TEST_CASE("remote launch preserves tracked aggregate pointers and mirrors local ones") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  std::array<uint8_t, 8> local{};
  auto remote = remoteAlloc(&context, local.size());
  auto localPointer = reinterpret_cast<uintptr_t>(local.data());
  const std::array<uint8_t, 2> types{static_cast<uint8_t>(Type::Ptr), static_cast<uint8_t>(Type::Ptr)};
  const std::array<void *, 2> arguments{&remote, &localPointer};
  const std::array<size_t, 2> mirrorSizes{local.size(), local.size()};
  const std::array<uint8_t, 2> mirrorKinds{2, 2};
  REQUIRE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                          arguments.data(), mirrorSizes.data(), mirrorKinds.data(), nullptr));
  CHECK(device.stub->remoteAllocs == 2);
  CHECK(device.stub->remoteFrees == 1);
  CHECK(context.remoteAllocations.size() == 1);
  REQUIRE(polyrt_remote_free(&context, remote));
  CHECK(device.stub->remoteFrees == 2);
  CHECK(context.remoteAllocations.empty());
}

TEST_CASE("remote launch copies a mutated local aggregate back from its mirror") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  queue.mutateFirstPointer = true;
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  uint64_t local = 0;
  auto localPointer = reinterpret_cast<uintptr_t>(&local);
  const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::Ptr)};
  const std::array<void *, 1> arguments{&localPointer};
  const std::array<size_t, 1> mirrorSizes{sizeof(local)};
  const std::array<uint8_t, 1> mirrorKinds{2};
  REQUIRE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                          arguments.data(), mirrorSizes.data(), mirrorKinds.data(), nullptr));
  CHECK(local == 0x123456789ABCDEF0ull);
}

TEST_CASE("remote launch passes a capture pointer into device memory through unchanged") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  queue.mutateFirstPointer = true;
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  auto remote = remoteAlloc(&context, sizeof(uint64_t));
  const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::Ptr)};
  const std::array<void *, 1> arguments{&remote};
  const std::array<size_t, 1> mirrorSizes{sizeof(uint64_t)};
  const std::array<uint8_t, 1> mirrorKinds{3};
  const std::array<const polyregion::runtime::TypeLayout *, 1> layouts{nullptr};
  REQUIRE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                          arguments.data(), mirrorSizes.data(), mirrorKinds.data(), layouts.data()));
  CHECK(device.stub->remoteAllocs == 1);
  CHECK(*reinterpret_cast<uint64_t *>(remote) == 0x123456789ABCDEF0ull);
  REQUIRE(polyrt_remote_free(&context, remote));
}

TEST_CASE("asserting remote modules take a leading error buffer and report the assert") {
  WithStubDevice device(false, false);
  WithStubManagedPlatform platform;
  StubQueue queue(std::chrono::seconds(1));
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  const unsigned char image = 0;
  std::array<const char *, 1> asserting{polyregion::conventions::ErrorArg};
  REQUIRE(polyrt_remote_load(&context, "asserting", static_cast<int32_t>(PlatformKind::Managed), static_cast<int32_t>(ModuleFormat::Object),
                             asserting.size(), asserting.data(), sizeof(image), &image));
  REQUIRE(polyrt_remote_load(&context, "plain", static_cast<int32_t>(PlatformKind::Managed), static_cast<int32_t>(ModuleFormat::Object), 0,
                             nullptr, sizeof(image), &image));
  int64_t n = 3;
  const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::IntS64)};
  const std::array<void *, 1> arguments{&n};

  REQUIRE(polyrt_remote_launch(&context, "plain", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(), arguments.data()));
  CHECK(queue.invokedTypes == std::vector{Type::IntS64, Type::Void});

  queue.raiseAssert = true;
  REQUIRE(polyrt_remote_launch(&context, "asserting", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(), arguments.data()));
  CHECK(queue.invokedTypes == std::vector{Type::Ptr, Type::IntS64, Type::Void});
  CHECK(assertReported());
  CHECK(device.stub->remoteAllocs == 1);
  CHECK(context.remoteAllocations.size() == 1);
}

TEST_CASE("asserting remote launches reuse one error block per context") {
  WithStubDevice device(false, false);
  WithStubManagedPlatform platform;
  StubQueue queue(std::chrono::seconds(1));
  const unsigned char image = 0;
  std::array<const char *, 1> asserting{polyregion::conventions::ErrorArg};
  int64_t n = 3;
  const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::IntS64)};
  const std::array<void *, 1> arguments{&n};
  {
    polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
    REQUIRE(polyrt_remote_load(&context, "asserting_reuse", static_cast<int32_t>(PlatformKind::Managed),
                               static_cast<int32_t>(ModuleFormat::Object), asserting.size(), asserting.data(), sizeof(image), &image));
    bool stale = false;
    queue.onInvoke = [&] {
      if (queue.raiseAssert) return;
      const auto *block = reinterpret_cast<const uint8_t *>(queue.firstPointer);
      stale |= std::any_of(block, block + polyregion::polyrt::assertBufferBytes, [](const uint8_t byte) { return byte != 0; });
    };
    for (size_t launch = 0; launch < 8; ++launch) {
      queue.raiseAssert = launch % 3 == 0;
      REQUIRE(polyrt_remote_launch(&context, "asserting_reuse", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                   arguments.data()));
    }
    CHECK_FALSE(stale);
    CHECK(device.stub->remoteAllocs == 1);
    CHECK(context.remoteAllocations.size() == 1);
  }
  CHECK(device.stub->remoteFrees == 1);
}

TEST_CASE("assert buffers of stdpar launches come back zeroed from the context pool") {
  WithStubDevice device(false, false);
  WithStubManagedPlatform platform;
  StubQueue queue(std::chrono::seconds(1));
  {
    polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
    const uint8_t reported[] = {7, 0, 0, 0, 'o', 'o', 'p', 's', 0};
    uintptr_t first = 0;
    for (size_t launch = 0; launch < 5; ++launch) {
      const auto block = polyregion::polyrt::allocAssertBuffer(context);
      if (launch == 0) first = block;
      CHECK(block == first);
      const auto *bytes = reinterpret_cast<const uint8_t *>(block);
      CHECK(std::all_of(bytes, bytes + polyregion::polyrt::assertBufferBytes, [](const uint8_t byte) { return byte == 0; }));
      const bool raise = launch % 2 == 0;
      if (raise) std::memcpy(reinterpret_cast<void *>(block), reported, sizeof(reported));
      const auto reason = polyregion::polyrt::readAssertBuffer(context, block);
      CHECK(reason.raised == raise);
      if (raise) CHECK(reason.message == "oops");
    }
    CHECK(device.stub->remoteAllocs == 1);
  }
  CHECK(device.stub->remoteFrees == 1);
}

TEST_CASE("remote launch settles copies in and out of mirrors with one wait each") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  std::array<uint8_t, 4> in{};
  std::array<uint8_t, 8> inOut{};
  auto inOutPointer = reinterpret_cast<uintptr_t>(inOut.data());
  const std::array<uint8_t, 2> types{static_cast<uint8_t>(Type::Ptr), static_cast<uint8_t>(Type::Ptr)};
  const std::array<void *, 2> arguments{in.data(), &inOutPointer};
  const std::array<size_t, 2> mirrorSizes{in.size(), inOut.size()};
  const std::array<uint8_t, 2> mirrorKinds{1, 2};
  REQUIRE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                          arguments.data(), mirrorSizes.data(), mirrorKinds.data(), nullptr));
  CHECK(queue.operations == "HHIWDW");
}

TEST_CASE("remote launch leaves a quiescent queue without waits") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  queue.idle = true;
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  std::array<uint8_t, 4> in{};
  std::array<uint8_t, 8> inOut{};
  auto inOutPointer = reinterpret_cast<uintptr_t>(inOut.data());
  const std::array<uint8_t, 2> types{static_cast<uint8_t>(Type::Ptr), static_cast<uint8_t>(Type::Ptr)};
  const std::array<void *, 2> arguments{in.data(), &inOutPointer};
  const std::array<size_t, 2> mirrorSizes{in.size(), inOut.size()};
  const std::array<uint8_t, 2> mirrorKinds{1, 2};
  REQUIRE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                          arguments.data(), mirrorSizes.data(), mirrorKinds.data(), nullptr));
  CHECK(queue.operations == "HHID");
}

TEST_CASE("remote sync follows blocking remote operations without waiting again") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  int64_t n = 3;
  const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::IntS64)};
  const std::array<void *, 1> arguments{&n};
  REQUIRE(polyrt_remote_launch(&context, "plain", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(), arguments.data()));
  polyrt_remote_sync(&context);
  auto remote = remoteAlloc(&context, 8);
  std::array<uint8_t, 8> local{};
  REQUIRE(polyrt_remote_memcpy(&context, remote, reinterpret_cast<uintptr_t>(local.data()), local.size(), 0));
  polyrt_remote_sync(&context);
  CHECK(queue.operations == "IWHW");
  REQUIRE(polyrt_remote_free(&context, remote));
}

TEST_CASE("remote launch preserves null aggregate pointers") {
  WithStubDevice device(false);
  StubQueue queue(std::chrono::seconds(1));
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  uintptr_t nullPointer = 0;
  const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::Ptr)};
  const std::array<void *, 1> arguments{&nullPointer};
  const std::array<size_t, 1> mirrorSizes{sizeof(uintptr_t)};
  const std::array<uint8_t, 1> mirrorKinds{2};
  REQUIRE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                          arguments.data(), mirrorSizes.data(), mirrorKinds.data(), nullptr));
  CHECK(device.stub->remoteAllocs == 0);
  CHECK(device.stub->remoteFrees == 0);
  CHECK(context.remoteAllocations.empty());
}

TEST_CASE("usm free returns a host fallback allocation to the host heap") {
  WithStubDevice device(false);
  auto *p = polyrt_usm_malloc(64);
  REQUIRE(p);
  polyrt_usm_free(p);
  CHECK(device.stub->sharedFrees == 0);
}

TEST_CASE("usm free returns a shared allocation to the device") {
  WithStubDevice device(true);
  auto *p = polyrt_usm_malloc(64);
  REQUIRE(p);
  polyrt_usm_free(p);
  CHECK(device.stub->sharedAllocs == 1);
  CHECK(device.stub->sharedFrees == 1);
}

TEST_CASE("usm free leaves a foreign pointer alone") {
  WithStubDevice device(true);
  auto *p = std::malloc(64);
  polyrt_usm_free(p);
  CHECK(device.stub->sharedFrees == 0);
  std::free(p);
}

TEST_CASE("usm one-past-end pointer resolves to the block that ends there") {
  WithStubDevice device(true);
  device.stub->packAllocations = true;
  auto *b = static_cast<int32_t *>(polyrt_usm_malloc(100 * sizeof(int32_t)));
  auto *ids = static_cast<int32_t *>(polyrt_usm_malloc(100 * sizeof(int32_t)));
  REQUIRE(b);
  REQUIRE(ids);
  const auto end = polyregion::rt_reflect::_rt_reflect_v(reinterpret_cast<uintptr_t>(b + 100));
  CHECK(end.size == 100 * sizeof(int32_t));
  CHECK(end.offset == 100 * sizeof(int32_t));
  CHECK(reinterpret_cast<uintptr_t>(b + 100) - end.offset == reinterpret_cast<uintptr_t>(b));
  CHECK(polyregion::rt_reflect::_rt_reflect_v(reinterpret_cast<uintptr_t>(ids)).offset == 0);
  polyrt_usm_free(ids);
  polyrt_usm_free(b);
}

TEST_CASE("adaptive JIT specialises hot values after loading generic code") {
  polyregion::polyrt::AdaptiveJitPolicy policy(3, 2);
  CHECK_FALSE(policy.select("kernel", 1).specialise);
  CHECK_FALSE(policy.select("kernel", 1).specialise);
  const auto admitted = policy.select("kernel", 1);
  CHECK(admitted.specialise);
  CHECK(admitted.admitted);
  CHECK(policy.select("kernel", 1).specialise);
  CHECK(policy.variantCount("kernel") == 1);
}

TEST_CASE("adaptive JIT caps variants and observations") {
  polyregion::polyrt::AdaptiveJitPolicy policy(2, 2, 3);
  CHECK_FALSE(policy.select("kernel", 1).specialise);
  CHECK(policy.select("kernel", 1).specialise);
  CHECK_FALSE(policy.select("kernel", 2).specialise);
  CHECK(policy.select("kernel", 2).specialise);
  CHECK_FALSE(policy.select("kernel", 3).specialise);
  CHECK_FALSE(policy.select("kernel", 3).specialise);
  CHECK(policy.variantCount("kernel") == 2);

  polyregion::polyrt::AdaptiveJitPolicy observations(8, 2, 3);
  for (uint64_t key = 0; key < 20; ++key)
    CHECK_FALSE(observations.select("kernel", key).specialise);
  CHECK(observations.observationCount("kernel") == 3);
}

template <typename T> constexpr size_t indirections() {
  if constexpr (std::is_pointer_v<T>) return 1 + indirections<std::remove_pointer_t<T>>();
  else return 0;
}

template <typename T> constexpr size_t componentSize() {
  if constexpr (std::is_pointer_v<T>) return componentSize<std::remove_pointer_t<T>>();
  else return sizeof(T);
}

struct StructWithStorage {
  TypeLayout s;
  std::unique_ptr<AggregateMember[]> storage;

  const TypeLayout &operator*() const { return s; }
  TypeLayout &operator*() { return s; }
};

template <typename T> StructWithStorage liftToStruct(const char *name, std::initializer_list<AggregateMember> members) {
  auto storage = std::make_unique<AggregateMember[]>(members.size());
  std::copy(members.begin(), members.end(), storage.get());

  TypeLayout s{
      .name = name,
      .sizeInBytes = sizeof(T),
      .alignmentInBytes = alignof(T),
      .attrs = LayoutAttrs::None,
      .memberCount = members.size(),
      .members = storage.get(),
  };

  return StructWithStorage{s, std::move(storage)};
}

#define NamedType(type) TypeLayout::named<type>(#type)
#define StructMember_(type_, member_, typePtr_)                                                                                            \
  AggregateMember {                                                                                                                        \
    .name = #member_,                                               /**/                                                                   \
        .offsetInBytes = offsetof(type_, member_),                  /**/                                                                   \
        .sizeInBytes = sizeof(type_::member_),                      /**/                                                                   \
        .ptrIndirection = indirections<decltype(type_::member_)>(), /**/                                                                   \
        .componentSize = componentSize<decltype(type_::member_)>(), /**/                                                                   \
        .type = typePtr_,                                           /**/                                                                   \
        .readOnly = 0,                                              /**/                                                                   \
        .resolvePtrSizeInBytes = nullptr                                                                                                   \
  }
#define Struct_(type_, ...) liftToStruct<type_>(#type_, {__VA_ARGS__})

const static TypeLayout floatType = NamedType(float);
const static TypeLayout int32Type = NamedType(int32_t);
const static TypeLayout int64Type = NamedType(int64_t);
const static TypeLayout doubleType = NamedType(double);

struct Fixture {
  std::unordered_map<uintptr_t, size_t> localAllocations, remoteAllocations;
  std::unordered_set<uintptr_t> heapAllocations, readOnlyAllocations, releasedRemote;
  size_t remoteWrites = 0, remoteReads = 0, remoteWriteBytes = 0, reflectQueries = 0;
  // stretches device-to-host copies the way a real transfer does, page by page
  std::chrono::microseconds remoteReadPause{0};
  std::function<void(void *)> duringRemoteRead;

  using QueryPtr = std::function<polyregion::polyrt::PtrQuery(const void *)>;
  using AllocateRemote = std::function<uintptr_t(size_t)>;
  using ReadRemote = std::function<void(void *, uintptr_t, size_t, size_t)>;
  using WriteRemote = std::function<void(const void *, uintptr_t, size_t, size_t)>;
  using FreeRemote = std::function<void(uintptr_t)>;
  polyregion::polyrt::SynchronisedMemAllocation<QueryPtr, AllocateRemote, ReadRemote, WriteRemote, FreeRemote> allocation;

  Fixture()
      : allocation(
            [&](const void *ptr) -> polyregion::polyrt::PtrQuery {
              reflectQueries++;
              const uintptr_t localPtr = reinterpret_cast<uintptr_t>(ptr);
              if (const auto it = localAllocations.find(localPtr); it != localAllocations.end()) {
                return polyregion::polyrt::PtrQuery{.sizeInBytes = it->second,
                                                    .offsetInBytes = 0,
                                                    .hostReadOnly = readOnlyAllocations.count(localPtr) != 0,
                                                    .heap = heapAllocations.count(localPtr) != 0};
              }
              for (auto [p, size] : localAllocations) {
                if (localPtr >= static_cast<uintptr_t>(p) && localPtr < static_cast<uintptr_t>(p) + size) {
                  return polyregion::polyrt::PtrQuery{
                      .sizeInBytes = size, .offsetInBytes = localPtr - static_cast<uintptr_t>(p), .heap = heapAllocations.count(p) != 0};
                }
              }
              return polyregion::polyrt::PtrQuery{0, 0};
            }, //
            [&](size_t size) {
              auto p = std::malloc(size);
              remoteAllocations.insert_or_assign(reinterpret_cast<uintptr_t>(p), size);
              releasedRemote.erase(reinterpret_cast<uintptr_t>(p));
              return reinterpret_cast<uintptr_t>(p);
            }, //
            [&](void *dst, uintptr_t src, size_t srcOffset, size_t size) {
              remoteReads++;
              if (duringRemoteRead) duringRemoteRead(dst);
              const size_t chunk = remoteReadPause.count() ? 4096 : size;
              for (size_t done = 0; done < size; done += chunk) {
                std::memcpy(static_cast<char *>(dst) + done, reinterpret_cast<char *>(src) + srcOffset + done,
                            std::min(chunk, size - done));
                if (remoteReadPause.count()) std::this_thread::sleep_for(remoteReadPause);
              }
            }, //
            [&](const void *src, uintptr_t dst, size_t dstOffset, size_t size) {
              remoteWrites++;
              remoteWriteBytes += size;
              return std::memcpy(reinterpret_cast<char *>(dst) + dstOffset, src, size);
            }, //
            [&](uintptr_t remotePtr) {
              releasedRemote.insert(remotePtr);
              std::free(reinterpret_cast<void *>(remotePtr));
            },
            false) {}

  template <typename T> T *mallocLocal(size_t count = 1) {
    auto p = std::malloc(sizeof(T) * count);
    localAllocations.emplace(reinterpret_cast<uintptr_t>(p), sizeof(T) * count);
    return static_cast<T *>(p);
  };

  template <typename T> T *mallocLocalHeap(size_t count) {
    auto p = mallocLocal<T>(count);
    heapAllocations.emplace(reinterpret_cast<uintptr_t>(p));
    return p;
  };

  template <typename T> std::enable_if_t<!std::is_pointer_v<T>, T> localToRemote(const T &t, const TypeLayout &s) {
    return *reinterpret_cast<T *>(allocation.syncLocalToRemote(&t, s));
  }
  template <typename T> T *localToRemote(const T *t, const TypeLayout &s) {
    return reinterpret_cast<T *>(allocation.syncLocalToRemote(t, s));
  }

  template <typename T> std::optional<uintptr_t> localToRemote(const T *t) { return allocation.syncLocalToRemote(t); }

  template <typename T> std::optional<uintptr_t> remoteToLocal(T *t) { return allocation.syncRemoteToLocal(t); }

  ~Fixture() {
    for (auto &[k, size] : localAllocations) {
      allocation.forget(reinterpret_cast<void *>(k), size, false);
      std::free(reinterpret_cast<void *>(k));
    }
    for (auto &[k, _] : remoteAllocations)
      if (!releasedRemote.count(k)) std::free(reinterpret_cast<void *>(k));
  }

  bool remoteLive(const void *p) const {
    const auto addr = reinterpret_cast<uintptr_t>(p);
    return std::any_of(remoteAllocations.begin(), remoteAllocations.end(), [&](const auto &entry) {
      return !releasedRemote.count(entry.first) && addr >= entry.first && addr < entry.first + entry.second;
    });
  }
};

TEST_CASE("generated mirror refreshes cached allocations between launches") {
  Fixture fixture;
  auto x = fixture.mallocLocal<int32_t>();
  *x = 41;

  struct Capture {
    int32_t *x;
  } capture{x};

  fixture.allocation.mirrorAlloc(&capture, sizeof(capture), false);
  auto remote = reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(x));
  CHECK(fixture.remoteWrites == 2);
  (*remote)++;
  fixture.allocation.unmirrorVisitClear();
  fixture.allocation.unmirrorReadAlloc(x);
  CHECK(*x == 42);
  fixture.allocation.disassociate(&capture, false);

  *x = 6;
  fixture.allocation.mirrorAlloc(&capture, sizeof(capture), false);
  CHECK(reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(x)) == remote);
  CHECK(*remote == 6);
  CHECK(fixture.remoteWrites == 4);
  CHECK(reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(x)) == remote);
  CHECK(fixture.remoteWrites == 4);
}

TEST_CASE("linked package mirror refreshes cached allocations between launches") {
  Fixture fixture;
  auto x = fixture.mallocLocal<int32_t>();
  *x = 41;

  const auto launch = [&] {
    fixture.allocation.unmirrorVisitClear();
    auto remote = reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(x));
    (*remote)++;
    fixture.allocation.unmirrorVisitClear();
    fixture.allocation.unmirrorReadAlloc(x);
    return remote;
  };

  auto remote = launch();
  CHECK(*x == 42);
  *x = 6;
  CHECK(launch() == remote);
  CHECK(*x == 7);
}

TEST_CASE("repeated launches upload only the mirror bytes the host changed") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  Fixture fixture;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  std::vector<int32_t *> blocks;
  for (int32_t i = 0; i < 4; ++i) {
    blocks.emplace_back(fixture.mallocLocalHeap<int32_t>(count));
    std::fill(blocks.back(), blocks.back() + count, i);
  }
  for (int32_t i = 0; i < 3; ++i) {
    blocks.emplace_back(fixture.mallocLocal<int32_t>(256));
    std::fill(blocks.back(), blocks.back() + 256, 10 + i);
  }
  struct Capture {
    int32_t *blocks[7];
  } capture{};

  const auto launch = [&](const auto &kernel) {
    const auto before = fixture.remoteWrites;
    fixture.allocation.mirrorAlloc(&capture, sizeof(capture), false);
    std::vector<int32_t *> remotes;
    for (auto *block : blocks)
      remotes.emplace_back(reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(block)));
    kernel(remotes);
    fixture.allocation.unmirrorVisitClear();
    for (auto *block : blocks)
      fixture.allocation.unmirrorReadAlloc(block);
    fixture.allocation.disassociate(&capture, false);
    return fixture.remoteWrites - before;
  };
  const auto idle = [](const std::vector<int32_t *> &) {};

  launch(idle);
  CHECK(launch(idle) == 1);
  CHECK(launch(idle) == 1);

  blocks[1][0] = 5;
  CHECK(launch(idle) == 2);
  blocks[2][count - 1] = 6;
  CHECK(launch(idle) == 2);
  blocks[5][100] = 7;
  CHECK(launch(idle) == 2);
  CHECK(launch(idle) == 1);

  launch([&](const std::vector<int32_t *> &remotes) {
    remotes[0][0] = 21;
    remotes[0][count - 1] = 22;
    remotes[6][5] = 23;
  });
  CHECK(blocks[0][0] == 21);
  CHECK(blocks[0][count - 1] == 22);
  CHECK(blocks[6][5] == 23);
  CHECK(launch([&](const std::vector<int32_t *> &remotes) {
          CHECK(remotes[0][0] == 21);
          CHECK(remotes[0][count - 1] == 22);
          CHECK(remotes[6][5] == 23);
        })
        == 1);

  blocks[0][0] = 0;
  CHECK(launch([&](const std::vector<int32_t *> &remotes) { CHECK(remotes[0][0] == 0); }) == 2);
}

TEST_CASE("read-only mirror blocks are uploaded once") {
  Fixture fixture;
  auto *xs = fixture.mallocLocal<int32_t>(256);
  std::fill(xs, xs + 256, 3);
  fixture.readOnlyAllocations.emplace(reinterpret_cast<uintptr_t>(xs));
  const auto launch = [&] {
    const auto before = fixture.remoteWrites;
    fixture.allocation.unmirrorVisitClear();
    auto *remote = reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(xs));
    CHECK(remote[7] == xs[7]);
    fixture.allocation.unmirrorVisitClear();
    fixture.allocation.unmirrorReadAlloc(xs);
    return fixture.remoteWrites - before;
  };
  CHECK(launch() == 1);
  CHECK(launch() == 1);
  CHECK(launch() == 0);
  CHECK(launch() == 0);
  xs[7] = 4;
  CHECK(launch() == 1);
  CHECK(launch() == 0);
}

TEST_CASE("stats count the host transfers of the shared runtime") {
  WithStubDevice device(false);
  WithStubManagedPlatform platform;
  auto &queue = static_cast<StubQueue &>(*polyregion::polyrt::currentQueue);
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  auto remote = remoteAlloc(&context, 16);
  std::array<char, 16> local{};
  const auto stats = [&] {
    PolyrtStats value{};
    REQUIRE(polyrt_stats(&context, &value));
    return value;
  };
  polyregion::polyrt::sma::stdRemoteWrite()(local.data(), remote, 0, local.size());
  polyregion::polyrt::sma::stdRemoteRead()(local.data(), remote, 0, local.size());
  REQUIRE(polyrt_remote_memcpy(&context, remote, reinterpret_cast<uintptr_t>(local.data()), local.size(), 0));
  REQUIRE(polyrt_remote_memcpy(&context, reinterpret_cast<uintptr_t>(local.data()), remote, local.size(), 1));
  CHECK(stats().h2dCopies == 2);
  CHECK(stats().d2hCopies == 2);
  CHECK(stats().waits == 4);
  queue.idle = true;
  polyregion::polyrt::sma::stdRemoteWrite()(local.data(), remote, 0, local.size());
  polyregion::polyrt::sma::stdRemoteRead()(local.data(), remote, 0, local.size());
  CHECK(stats().h2dCopies == 3);
  CHECK(stats().d2hCopies == 3);
  CHECK(stats().waits == 4);
  REQUIRE(polyrt_remote_free(&context, remote));
}

TEST_CASE("resident heap blocks fault their device copy back and upload again only after a host write") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  Fixture fixture;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  auto xs = fixture.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  struct Capture {
    int32_t *xs;
  } capture{xs};

  const auto launch = [&](const auto &kernel) {
    fixture.allocation.mirrorAlloc(&capture, sizeof(capture), false);
    auto remote = reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(xs));
    kernel(remote);
    fixture.allocation.unmirrorVisitClear();
    fixture.allocation.unmirrorReadAlloc(xs);
    fixture.allocation.disassociate(&capture, false);
    return remote;
  };

  auto remote = launch([&](int32_t *r) { std::fill(r, r + count, 2); });
  const auto readsAfterLaunch = fixture.remoteReads;
  CHECK(xs[0] == 2);
  CHECK(fixture.remoteReads == readsAfterLaunch);
  CHECK(xs[count - 1] == 2);
  CHECK(xs[count / 2] == 2);
  CHECK(fixture.remoteReads == readsAfterLaunch + 1);

  const auto writesBeforeRelaunch = fixture.remoteWrites;
  CHECK(launch([&](int32_t *r) { r[count / 2] = 3; }) == remote);
  CHECK(fixture.remoteWrites - writesBeforeRelaunch <= 3);
  CHECK(xs[count / 2] == 3);

  xs[count / 2 + 1] = 7;
  launch([&](int32_t *r) { CHECK(r[count / 2 + 1] == 7); });
  CHECK(xs[count / 2 + 1] == 7);

  fixture.allocation.disassociate(xs, false);
  xs[count / 2] = 9;
  CHECK(xs[count / 2] == 9);
}

static bool residentCopy(const size_t bytes, const size_t pages, const size_t fullBytes) {
  return polyregion::polyrt::guard::AtomicPublish ? bytes <= pages * polyregion::polyrt::guard::pageSize() : bytes >= fullBytes;
}

TEST_CASE("typed read-back of a pointer-free array first mirrored without a layout queries a bounded number of records") {
  struct Capture {
    double *xs;
  };
  auto captureMeta = Struct_(Capture, StructMember_(Capture, xs, &doubleType));

  Fixture fixture;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(double);
  auto xs = fixture.mallocLocalHeap<double>(count);
  std::fill(xs, xs + count, 1.0);
  Capture capture{xs};

  fixture.allocation.unmirrorVisitClear();
  auto remote = reinterpret_cast<double *>(fixture.allocation.mirrorEnsure(xs));
  std::fill(remote, remote + count, 2.0);
  fixture.allocation.unmirrorVisitClear();
  fixture.allocation.unmirrorReadAlloc(xs);
  CHECK(xs[count - 1] == 2.0);

  const auto remoteCapture = fixture.localToRemote(&capture, *captureMeta);
  REQUIRE(remoteCapture->xs == remote);
  std::fill(remote, remote + count, 3.0);
  fixture.allocation.unmirrorVisitClear();
  fixture.reflectQueries = 0;
  fixture.allocation.syncRemoteToLocalAfterLaunch(&capture);
  CHECK(fixture.reflectQueries <= 8);
  fixture.allocation.disassociate(&capture, false);

  const auto relaunch = [&](const auto &kernel) {
    fixture.allocation.unmirrorVisitClear();
    const auto before = fixture.remoteWriteBytes;
    auto r = reinterpret_cast<double *>(fixture.allocation.mirrorEnsure(xs));
    kernel(r);
    fixture.allocation.unmirrorVisitClear();
    fixture.allocation.unmirrorReadAlloc(xs);
    return fixture.remoteWriteBytes - before;
  };
  CHECK(residentCopy(relaunch([&](double *r) { CHECK(r[count / 2] == 3.0); }), 4, count * sizeof(double)));
  CHECK(xs[0] == 3.0);
  CHECK(xs[count - 1] == 3.0);

  xs[count / 2] = 9.0;
  CHECK(relaunch([&](double *r) { CHECK(r[count / 2] == 9.0); }) >= count * sizeof(double) / 2);
  CHECK(residentCopy(relaunch([&](double *r) { CHECK(r[count / 2] == 9.0); }), 4, count * sizeof(double)));
  fixture.allocation.disassociate(xs, false);
}

TEST_CASE("reading back a typed pointer-free array directly after a launch settles it on the device") {
  struct Capture {
    double *xs;
  };
  auto captureMeta = Struct_(Capture, StructMember_(Capture, xs, &doubleType));

  Fixture fixture;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(double);
  auto xs = fixture.mallocLocalHeap<double>(count);
  std::fill(xs, xs + count, 1.0);
  Capture capture{xs};

  const auto launch = [&](const double value) {
    fixture.allocation.unmirrorVisitClear();
    const auto before = fixture.remoteWriteBytes;
    const auto remoteCapture = fixture.localToRemote(&capture, *captureMeta);
    const auto bytes = fixture.remoteWriteBytes - before;
    std::fill(remoteCapture->xs, remoteCapture->xs + count, value);
    fixture.allocation.unmirrorVisitClear();
    fixture.allocation.syncRemoteToLocalAfterLaunch(xs);
    fixture.allocation.disassociate(&capture, false);
    return bytes;
  };

  launch(2.0);
  CHECK(xs[count - 1] == 2.0);
  CHECK(residentCopy(launch(3.0), 8, count * sizeof(double)));
  CHECK(xs[count / 2] == 3.0);

  xs[count / 2] = 9.0;
  fixture.allocation.unmirrorVisitClear();
  const auto remoteCapture = fixture.localToRemote(&capture, *captureMeta);
  CHECK(remoteCapture->xs[count / 2] == 9.0);
  fixture.allocation.disassociate(&capture, false);
  fixture.allocation.disassociate(xs, false);
}

TEST_CASE("a pointer-bearing view of a record settled as pointer-free restores pointer translation") {
  struct Holder {
    int32_t *p;
    int32_t *q;
  };
  struct Plain {
    double *d;
  };
  struct Typed {
    Holder *h;
  };
  auto holderMeta = Struct_(Holder, StructMember_(Holder, p, &int32Type), StructMember_(Holder, q, &int32Type));
  auto plainMeta = Struct_(Plain, StructMember_(Plain, d, &doubleType));
  auto typedMeta = Struct_(Typed, StructMember_(Typed, h, &*holderMeta));

  Fixture fixture;
  auto ys = fixture.mallocLocal<int32_t>(4);
  std::fill(ys, ys + 4, 5);
  auto block = new (fixture.mallocLocal<Holder>()) Holder{ys, ys + 2};

  fixture.allocation.unmirrorVisitClear();
  fixture.allocation.mirrorEnsure(block, sizeof(double), true);
  Plain plain{reinterpret_cast<double *>(block)};
  fixture.localToRemote(&plain, *plainMeta);
  fixture.allocation.disassociate(&plain, false);

  Typed typed{block};
  const auto remoteTyped = fixture.localToRemote(&typed, *typedMeta);
  REQUIRE(remoteTyped->h != block);
  CHECK(remoteTyped->h->p != ys);
  CHECK(remoteTyped->h->q == remoteTyped->h->p + 2);
  CHECK(*remoteTyped->h->p == 5);
}

TEST_CASE("resident heap blocks reclaim before code that bypasses the fault path") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  Fixture fixture;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  auto xs = fixture.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  struct Capture {
    int32_t *xs;
  } capture{xs};

  fixture.allocation.mirrorAlloc(&capture, sizeof(capture), false);
  auto remote = reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(xs));
  std::fill(remote, remote + count, 4);
  fixture.allocation.unmirrorVisitClear();
  fixture.allocation.unmirrorReadAlloc(xs);
  fixture.allocation.disassociate(&capture, false);

  fixture.allocation.reclaimRange(xs + count / 2, sizeof(int32_t) * 4, /*forWrite*/ true);
  std::vector<int32_t> copied(xs, xs + count);
  CHECK(std::all_of(copied.begin(), copied.end(), [](const int32_t x) { return x == 4; }));
  fixture.allocation.disassociate(xs, false);
}

TEST_CASE("freeing an unguarded block never waits on a device read in flight") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  Fixture fixture;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  auto xs = fixture.mallocLocalHeap<int32_t>(count);
  auto unrelated = fixture.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  struct Capture {
    int32_t *xs;
  } capture{xs};

  std::future<void> freed;
  bool freedDuringRead = false;
  fixture.duringRemoteRead = [&](void *dst) {
    if (freed.valid() || dst < xs || dst >= xs + count) return;
    freed = std::async(std::launch::async, [&] { fixture.allocation.forget(unrelated, count * sizeof(int32_t), false); });
    freedDuringRead = freed.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  };
  fixture.allocation.mirrorAlloc(&capture, sizeof(capture), false);
  auto remote = reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(xs));
  std::fill(remote, remote + count, 2);
  fixture.allocation.unmirrorVisitClear();
  fixture.allocation.unmirrorReadAlloc(xs);
  fixture.allocation.disassociate(&capture, false);
  fixture.duringRemoteRead = {};

  REQUIRE(freed.valid());
  CHECK(freedDuringRead);
  freed.wait();
  CHECK(xs[count / 2] == 2);
  fixture.allocation.disassociate(xs, false);
}

TEST_CASE("resident heap blocks resolve concurrent host reads from many threads") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  Fixture fixture;
  fixture.remoteReadPause = std::chrono::microseconds(50);
  constexpr size_t blocks = 8, threads = 8;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  struct Capture {
    int32_t *xs[blocks];
  } capture{};
  for (auto &xs : capture.xs) {
    xs = fixture.mallocLocalHeap<int32_t>(count);
    std::fill(xs, xs + count, 0);
  }

  fixture.allocation.mirrorAlloc(&capture, sizeof(capture), false);
  for (size_t b = 0; b < blocks; ++b) {
    auto remote = reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(capture.xs[b]));
    std::fill(remote, remote + count, static_cast<int32_t>(b + 1));
  }
  fixture.allocation.unmirrorVisitClear();
  for (auto xs : capture.xs)
    fixture.allocation.unmirrorReadAlloc(xs);
  fixture.allocation.disassociate(&capture, false);

  const auto readsAfterLaunch = fixture.remoteReads;
  std::atomic<size_t> wrong{0};
  std::vector<std::thread> readers;
  for (size_t t = 0; t < threads; ++t)
    readers.emplace_back([&, t] {
      for (size_t k = 0; k < blocks; ++k) {
        const size_t b = (k + t) % blocks;
        const int32_t *xs = capture.xs[b];
        for (size_t i = 0; i < count; ++i)
          if (xs[i] != static_cast<int32_t>(b + 1)) wrong.fetch_add(1);
      }
    });
  for (auto &reader : readers)
    reader.join();
  CHECK(wrong.load() == 0);
  CHECK(fixture.remoteReads - readsAfterLaunch <= blocks);
  for (auto xs : capture.xs)
    fixture.allocation.disassociate(xs, false);
}

#ifndef _WIN32
TEST_CASE("record wrappers reclaim guarded blocks before libc or the kernel touches them") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  static Fixture *active = nullptr;
  Fixture fixture;
  active = &fixture;
  polyregion::rt_reflect::_rt_set_reclaim_cb(
      +[](const void *p, const size_t n, const bool forWrite) { active->allocation.reclaimRange(p, n, forWrite); });
  polyregion::rt_reflect::_rt_set_release_cb(+[](void *p, const size_t n) { active->allocation.forget(p, n, false); });

  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t), bytes = count * sizeof(int32_t);
  auto xs = fixture.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  struct Capture {
    int32_t *xs;
  } capture{xs};
  const auto launch = [&](const auto &kernel) {
    fixture.allocation.mirrorAlloc(&capture, sizeof(capture), false);
    auto remote = reinterpret_cast<int32_t *>(fixture.allocation.mirrorEnsure(capture.xs));
    kernel(remote);
    fixture.allocation.unmirrorVisitClear();
    fixture.allocation.unmirrorReadAlloc(capture.xs);
    fixture.allocation.disassociate(&capture, false);
  };
  const auto deviceFill = [&](const int32_t value) { launch([&](int32_t *r) { std::fill(r, r + count, value); }); };
  const auto all = [&](const int32_t *ys, const int32_t value) {
    return std::all_of(ys, ys + count, [&](const int32_t y) { return y == value; });
  };
  FILE *file = std::tmpfile();
  REQUIRE(file);
  const int fd = fileno(file);
  std::vector<int32_t> onDisk(count);

  deviceFill(5);
  CHECK(polyrt_record_write(fd, xs, bytes) == static_cast<ssize_t>(bytes));
  REQUIRE(::pread(fd, onDisk.data(), bytes, 0) == static_cast<ssize_t>(bytes));
  CHECK(all(onDisk.data(), 5));

  deviceFill(6);
  CHECK(polyrt_record_pread(fd, xs, bytes, 0) == static_cast<ssize_t>(bytes));
  CHECK(all(xs, 5));
  launch([&](int32_t *r) { CHECK(all(r, 5)); });

  deviceFill(7);
  std::rewind(file);
  CHECK(polyrt_record_fwrite(xs, sizeof(int32_t), count, file) == count);
  std::fflush(file);
  REQUIRE(::pread(fd, onDisk.data(), bytes, 0) == static_cast<ssize_t>(bytes));
  CHECK(all(onDisk.data(), 7));

  deviceFill(8);
  std::rewind(file);
  CHECK(polyrt_record_fread(xs, sizeof(int32_t), count, file) == count);
  CHECK(all(xs, 7));

  deviceFill(9);
  auto grown = static_cast<int32_t *>(polyrt_record_realloc(xs, bytes * 2));
  REQUIRE(grown);
  fixture.localAllocations.erase(reinterpret_cast<uintptr_t>(xs));
  fixture.heapAllocations.erase(reinterpret_cast<uintptr_t>(xs));
  CHECK(all(grown, 9));

  std::fclose(file);
  polyrt_record_free(grown);
  CHECK(polyregion::rt_reflect::_rt_reflect_v(reinterpret_cast<uintptr_t>(grown)).size == 0);
  polyregion::rt_reflect::_rt_set_reclaim_cb(nullptr);
  polyregion::rt_reflect::_rt_set_release_cb(nullptr);
  active = nullptr;
}
#endif

TEST_CASE("runtime mirror refreshes cached allocations between launches") {
  struct Capture {
    int32_t *x;
  };
  auto captureMeta = Struct_(Capture, StructMember_(Capture, x, &int32Type));

  Fixture fixture;
  auto x = fixture.mallocLocal<int32_t>();
  *x = 41;
  Capture capture{x};

  auto remoteCapture = fixture.localToRemote(&capture, *captureMeta);
  auto remote = remoteCapture->x;
  (*remote)++;
  fixture.remoteToLocal(&capture);
  CHECK(*x == 42);
  fixture.allocation.disassociate(&capture, false);

  *x = 6;
  remoteCapture = fixture.localToRemote(&capture, *captureMeta);
  CHECK(remoteCapture->x == remote);
  CHECK(*remote == 6);
}

TEST_CASE("runtime mirror translates captured pointer slots back after a launch") {
  struct Capture {
    int32_t **slot;
  };
  auto captureMeta = Struct_(Capture, StructMember_(Capture, slot, &int32Type));

  Fixture fixture;
  auto xs = fixture.mallocLocal<int32_t>(4);
  std::fill(xs, xs + 4, 1);
  auto slot = fixture.mallocLocal<int32_t *>();
  *slot = xs;
  Capture capture{slot};

  auto remoteCapture = fixture.localToRemote(&capture, *captureMeta);
  int32_t *remoteXs = *remoteCapture->slot;
  CHECK(remoteXs != xs);
  std::fill(remoteXs, remoteXs + 4, 2);
  fixture.allocation.syncRemoteToLocalAfterLaunch(&capture);
  CHECK(*slot == xs);
  CHECK(xs[3] == 2);
}

TEST_CASE("runtime mirror preserves an interior offset when refreshing a reused allocation") {
  struct Pair {
    int32_t first;
    int32_t second;
  };
  struct Capture {
    int32_t *result;
  };
  auto pairMeta = Struct_(Pair, StructMember_(Pair, first, &int32Type), StructMember_(Pair, second, &int32Type));
  auto captureMeta = Struct_(Capture, StructMember_(Capture, result, &int32Type));

  Fixture fixture;
  auto pair = new (fixture.mallocLocal<Pair>()) Pair{3, 4};
  auto remotePair = fixture.localToRemote(pair, *pairMeta);

  pair->second = 9;
  REQUIRE(fixture.allocation.invalidateLocal(&pair->second) == std::optional{reinterpret_cast<uintptr_t>(&remotePair->second)});
  CHECK(fixture.remoteToLocal(&pair->second) == std::optional{reinterpret_cast<uintptr_t>(&remotePair->second)});
  CHECK(pair->second == 9);
  Capture capture{&pair->second};
  auto remoteCapture = fixture.localToRemote(&capture, *captureMeta);

  CHECK(remoteCapture->result == &remotePair->second);
  CHECK(remotePair->first == 3);
  CHECK(remotePair->second == 9);
}

TEST_CASE("freeing a block drops mirrors keyed at its interior pointers") {
  struct Capture {
    int32_t *data;
  };
  auto captureMeta = Struct_(Capture, StructMember_(Capture, data, &int32Type));

  Fixture fixture;
  constexpr size_t overAlign = 32, count = 64, blockBytes = overAlign + count * sizeof(int32_t);
  auto region = fixture.mallocLocalHeap<char>(blockBytes);
  const auto regionKey = reinterpret_cast<uintptr_t>(region);
  auto stale = reinterpret_cast<int32_t *>(region + overAlign);
  std::fill(stale, stale + count, 1);
  Capture staleCapture{stale};
  auto remoteStale = fixture.localToRemote(&staleCapture, *captureMeta)->data;
  fixture.allocation.disassociate(&staleCapture, false);

  fixture.localAllocations.erase(regionKey);
  fixture.heapAllocations.erase(regionKey);
  fixture.allocation.forget(region, blockBytes, false);

  fixture.localAllocations.emplace(regionKey, blockBytes);
  fixture.heapAllocations.emplace(regionKey);
  auto reused = reinterpret_cast<int32_t *>(region);
  std::fill(reused, reused + blockBytes / sizeof(int32_t), 2);
  Capture reusedCapture{reused};
  auto remoteReused = fixture.localToRemote(&reusedCapture, *captureMeta)->data;
  fixture.allocation.disassociate(&reusedCapture, false);

  CHECK(remoteReused != remoteStale);
  CHECK(fixture.remoteToLocal(stale) == std::optional{reinterpret_cast<uintptr_t>(remoteReused + overAlign / sizeof(int32_t))});
  remoteReused[overAlign / sizeof(int32_t)] = 3;
  fixture.allocation.syncRemoteToLocal(stale);
  CHECK(stale[0] == 3);
}

TEST_CASE("ptr-indirect-2-star") {
  struct Foo {
    float **a;
  };
  auto fooMeta = Struct_(Foo, StructMember_(Foo, a, &floatType));
  Fixture fixture;
  size_t N = 10;
  size_t M = 5;
  auto as = fixture.mallocLocal<float *>(N);
  for (size_t i = 0; i < N; ++i) {
    as[i] = fixture.mallocLocal<float>(M);
    for (size_t j = 0; j < M; ++j) {
      as[i][j] = j;
    }
  }
  auto local = new (fixture.mallocLocal<Foo>()) Foo{as};
  Foo *remote = fixture.localToRemote(local, *fooMeta);
  CHECK(&remote != &local);
  CHECK(local->a != remote->a);
  for (size_t i = 0; i < N; ++i) {
    CHECK(local->a[i] != remote->a[i]);
    for (size_t j = 0; j < M; ++j) {
      CHECK(local->a[i][j] == remote->a[i][j]);
    }
  }

  remote->a[N - 1][M - 1] = 42;
  CHECK(local->a[N - 1][M - 1] == M - 1);
  CHECK(fixture.remoteToLocal(local->a[N - 1]) == std::optional{reinterpret_cast<uintptr_t>(remote->a[N - 1])});
  CHECK(local->a[N - 1][M - 1] == 42);

  local->a[N - 1][M - 1] = 43;

  CHECK(remote->a[N - 1][M - 1] == 42);
  CHECK(fixture.allocation.invalidateLocal(local->a[N - 1]) == std::optional{reinterpret_cast<uintptr_t>(remote->a[N - 1])});
  fixture.localToRemote(local->a[N - 1]);
  CHECK(remote->a[N - 1][M - 1] == 43);
}

TEST_CASE("ptr-indirect-nested-2-star") {
  struct check_array {
    int32_t **xs;
  };

  struct test_utils {
    int32_t *result;
    check_array f;
  };

  auto fooMeta = Struct_(check_array, StructMember_(check_array, xs, &int32Type));
  auto barMeta = Struct_(test_utils,                                    //
                         StructMember_(test_utils, result, &int32Type), //
                         StructMember_(test_utils, f, &*fooMeta), );

  Fixture fixture;
  size_t N = 10;
  const auto as = fixture.mallocLocal<int32_t>(N);
  std::iota(as, as + N, 0);

  const auto asRef = fixture.mallocLocal<int32_t *>();
  *asRef = as;

  const auto result = fixture.mallocLocal<int32_t>();
  *result = 42;

  auto local = new (fixture.mallocLocal<test_utils>()) test_utils{result, check_array{asRef}};

  test_utils *remote = fixture.localToRemote(local, *barMeta);

  (*barMeta).visualise(stderr);
  (*fooMeta).visualise(stderr);

  //
  // int p[1]={42};
  //
  // fprintf(stderr, "a = %p\n", p);
  // const auto ff = [=]()   {
  //   fprintf(stderr, "Lam: a = %p\n", p);
  // };
  // static_assert(sizeof(decltype(ff)) == sizeof(void*));
  //
  //
  // auto raw = reinterpret_cast<const char*>(&ff);
  // void* value=0;
  // std::memcpy(&value, raw, sizeof(value));
  // fprintf(stderr, "f = %p\n", value);
  //
  // fprintf(stderr, "f = %d !\n", *static_cast<int32_t*>(value));
}

TEST_CASE("ptr-indirect-3-star") {
  struct Bar {
    int32_t a;
  };
  struct Foo {
    Bar ***a;
  };
  auto barMeta = Struct_(Bar, StructMember_(Bar, a, &int32Type));
  auto fooMeta = Struct_(Foo, StructMember_(Foo, a, &*barMeta));
  Fixture fixture;
  auto bar = new (fixture.mallocLocal<Bar>()) Bar(42);
  auto barPtr = new (fixture.mallocLocal<Bar *>()) Bar *;
  auto barPtrPtr = new (fixture.mallocLocal<Bar **>()) Bar **;
  barPtr[0] = bar;
  barPtrPtr[0] = barPtr;
  Foo expected{barPtrPtr};
  Foo actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(expected.a != actual.a);
  CHECK(*expected.a != *actual.a);
  CHECK((*expected.a) != (*actual.a));
  CHECK((*(*expected.a)) != (*(*actual.a)));
  CHECK((*(*expected.a))->a == (*(*actual.a))->a);
}

TEST_CASE("simple") {
  struct Foo {
    int32_t a;
    int32_t b;
  };
  auto fooMeta = Struct_(Foo,                               //
                         StructMember_(Foo, a, &int32Type), //
                         StructMember_(Foo, b, &int32Type), //
  );
  Foo expected{42, 43};
  Fixture fixture;
  Foo actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(expected.a == actual.a);
  CHECK(expected.b == actual.b);
  CHECK(fixture.remoteAllocations.size() == 1);
}

TEST_CASE("simple-nested") {
  struct Foo {
    int32_t a;
    int32_t b;
  };

  struct Bar {
    Foo foo;
    int32_t c;
    int32_t d;
  };

  auto fooMeta = Struct_(Foo,                               //
                         StructMember_(Foo, a, &int32Type), //
                         StructMember_(Foo, b, &int32Type)  //
  );

  auto barMeta = Struct_(Bar,                                //
                         StructMember_(Bar, foo, &*fooMeta), //
                         StructMember_(Bar, c, &int32Type),  //
                         StructMember_(Bar, d, &int32Type)   //
  );

  Fixture fixture;
  auto expected = new (fixture.mallocLocal<Bar>()) Bar{{42, 43}, 44, 45};
  Bar *actual = fixture.localToRemote(expected, *barMeta);

  CHECK(actual != expected);
  CHECK(expected->foo.a == actual->foo.a);
  CHECK(expected->foo.b == actual->foo.b);
  CHECK(expected->c == actual->c);
  CHECK(expected->d == actual->d);
  CHECK(fixture.remoteAllocations.size() == 1);

  actual->foo.a = 0;
  actual->foo.b = 1;
  actual->c = 2;
  actual->d = 3;
  CHECK(fixture.remoteToLocal(expected) == std::optional{reinterpret_cast<uintptr_t>(actual)});
  CHECK(expected->foo.a == 0);
  CHECK(expected->foo.b == 1);
  CHECK(expected->c == 2);
  CHECK(expected->d == 3);
}

TEST_CASE("linkedlist") {
  struct Node {
    int32_t data;
    Node *next;
  };

  auto nodeMetaDeferred = Struct_(Node);
  auto nodeMeta = Struct_(Node,                                         //
                          StructMember_(Node, data, &int32Type),        //
                          StructMember_(Node, next, &*nodeMetaDeferred) //
  );
  (*nodeMetaDeferred).memberCount = (*nodeMeta).memberCount;
  (*nodeMetaDeferred).members = (*nodeMeta).members;

  Fixture fixture;

  auto node3 = new (fixture.mallocLocal<Node>()) Node{3, nullptr};
  auto node2 = new (fixture.mallocLocal<Node>()) Node{2, node3};
  auto node1 = new (fixture.mallocLocal<Node>()) Node{1, node2};
  auto actual = fixture.localToRemote(*node1, *nodeMeta);
  CHECK(&actual != node1);
  CHECK(actual.data == 1);
  CHECK(actual.next != node2);
  CHECK(actual.next->data == 2);
  CHECK(actual.next->next != node3);
  CHECK(actual.next->next->data == 3);
  CHECK(actual.next->next->next == nullptr);
  CHECK(fixture.remoteAllocations.size() == 3); // 3 allocations: 3 Nodes

  actual.next->next->data = 42;
  CHECK(fixture.remoteToLocal(node3) == std::optional{reinterpret_cast<uintptr_t>(actual.next->next)});

  CHECK(node3->next == nullptr);
  CHECK(node3->data == 42);
  CHECK(node2->next == node3);
  CHECK(node2->data == 2);
  CHECK(node1->next == node2);
  CHECK(node1->data == 1);
}

TEST_CASE("linkedlist-indirect") {
  struct Other;
  struct Node {
    int32_t data;
    Other *other;
  };

  struct Other {
    Node *value;
  };

  auto nodeMetaDeferred = Struct_(Node);
  auto otherMetaDeferred = Struct_(Other);
  auto otherMeta = Struct_(Other,                                          //
                           StructMember_(Other, value, &*nodeMetaDeferred) //
  );

  auto nodeMeta = Struct_(Node,                                           //
                          StructMember_(Node, data, &int32Type),          //
                          StructMember_(Node, other, &*otherMetaDeferred) //
  );
  (*nodeMetaDeferred).memberCount = (*nodeMeta).memberCount;
  (*nodeMetaDeferred).members = (*nodeMeta).members;
  (*otherMetaDeferred).memberCount = (*otherMeta).memberCount;
  (*otherMetaDeferred).members = (*otherMeta).members;

  Fixture fixture;
  auto node3 = new (fixture.mallocLocal<Node>()) Node{3, new (fixture.mallocLocal<Other>()) Other{nullptr}};
  auto node2 = new (fixture.mallocLocal<Node>()) Node{2, new (fixture.mallocLocal<Other>()) Other{node3}};
  auto node1 = new (fixture.mallocLocal<Node>()) Node{1, new (fixture.mallocLocal<Other>()) Other{node2}};

  auto remote = fixture.localToRemote(*node1, *nodeMeta);

  const char *p = reinterpret_cast<char *>(node1);
  (*nodeMeta).visualise(stderr, [&](size_t offset, const AggregateMember &m) {
    auto x = p + offset;
    std::fprintf(stderr, "value=");
    if (m.ptrIndirection != 0) {
      polyregion::compiletime::showPtr(stderr, sizeof(void *), x);
    } else {
      polyregion::compiletime::showInt(stderr, false, m.type->sizeInBytes, x);
    }
  });

  CHECK(&remote != node1);
  CHECK(remote.data == 1);
  CHECK(remote.other != node1->other);
  CHECK(remote.other->value != node2);
  CHECK(remote.other->value->data == 2);
  CHECK(remote.other->value->other->value != node3);
  CHECK(remote.other->value->other->value->data == 3);
  CHECK(remote.other->value->other->value->other->value == nullptr);
  CHECK(fixture.remoteAllocations.size() == 6); // 6 allocations: 3 Nodes + 3 Others
  CHECK(node3->other->value == nullptr);

  remote.other->value->other->value->data = 42;
  CHECK(fixture.remoteToLocal(node3) == std::optional{reinterpret_cast<uintptr_t>(remote.other->value->other->value)});

  CHECK(node3->other->value == nullptr);
  CHECK(node3->data == 42);
  CHECK(node2->other->value == node3);
  CHECK(node2->data == 2);
  CHECK(node1->other->value == node2);
  CHECK(node1->data == 1);
}

TEST_CASE("ptr") {
  struct Bar {
    int64_t x;
    float *a;
  };
  struct Foo {
    float *a;
    int32_t b;
    float *c;
    Bar *bar;
    Bar barOpaque;
  };
  auto barMeta = Struct_(Bar,                               //
                         StructMember_(Bar, x, &int64Type), //
                         StructMember_(Bar, a, &floatType), //
  );
  auto fooMeta = Struct_(Foo,                                     //
                         StructMember_(Foo, a, &floatType),       //
                         StructMember_(Foo, b, &int32Type),       //
                         StructMember_(Foo, c, &floatType),       //
                         StructMember_(Foo, bar, &*barMeta),      //
                         StructMember_(Foo, barOpaque, &*barMeta) //
  );

  Fixture fixture;
  int N = 10;
  auto as = fixture.mallocLocal<float>(N);
  for (int i = 0; i < N; ++i)
    as[i] = i;
  auto bar = new (fixture.mallocLocal<Bar>()) Bar{.x = 43, .a = as};
  Foo expected{.a = as, .b = 42, .c = nullptr, .bar = bar, .barOpaque = *bar};
  auto actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(actual.a != expected.a);
  CHECK(actual.b == expected.b);
  CHECK(actual.c == expected.c);
  CHECK(actual.bar != expected.bar);
  CHECK(actual.bar->x == expected.bar->x);
  CHECK(actual.bar->a != expected.bar->a);
  CHECK(actual.barOpaque.x == expected.barOpaque.x);
  CHECK(actual.barOpaque.a != expected.barOpaque.a);
  CHECK(std::memcmp(actual.a, expected.a, sizeof(float) * N) == 0);
  CHECK(fixture.remoteAllocations.size() == 3);
}

TEST_CASE("ptr-offset-simple") {
  struct Foo {
    float *a;
  };
  auto fooMeta = Struct_(Foo, StructMember_(Foo, a, &floatType));

  Fixture fixture;
  size_t N = 10;
  size_t Offset = 5;
  auto as = fixture.mallocLocal<float>(N);
  std::iota(as, as + N, 0);

  Foo expected{as + Offset};
  auto actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(actual.a != expected.a);

  CHECK(std::memcmp(actual.a, expected.a, sizeof(float) * (N - Offset)) == 0);
  CHECK(fixture.remoteAllocations.size() == 2);
}

TEST_CASE("ptr-offset-internal") {
  struct Foo {
    float *a;
    float *b;
  };
  auto fooMeta = Struct_(Foo,                               //
                         StructMember_(Foo, a, &floatType), //
                         StructMember_(Foo, b, &floatType)  //

  );
  Fixture fixture;
  size_t N = 10;
  size_t Offset = 5;
  auto as = fixture.mallocLocal<float>(N);
  std::iota(as, as + N, 0);

  Foo expected{as, as + Offset};
  auto actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(actual.a != expected.a);
  CHECK(actual.b != expected.b);

  CHECK(std::memcmp(actual.a, expected.a, sizeof(float) * N) == 0);

  CHECK(std::memcmp(actual.b, expected.b, sizeof(float) * (N - Offset)) == 0);
  CHECK(fixture.remoteAllocations.size() == 2); // struct and a*, no b*

  CHECK(expected.a[0] == 0);
  CHECK(expected.b[0] == 5);

  actual.a[0] = 42;
  actual.b[0] = 43;

  CHECK(fixture.remoteToLocal(expected.b) == std::optional{reinterpret_cast<uintptr_t>(actual.b)});
  CHECK(expected.a[0] == 0);
  CHECK(expected.b[0] == 43);

  CHECK(fixture.remoteToLocal(expected.a) == std::optional{reinterpret_cast<uintptr_t>(actual.a)});
  CHECK(expected.a[0] == 42);
  CHECK(expected.b[0] == 43);
}

TEST_CASE("ptr-offset-below-base") {
  struct Foo {
    float *a;
  };
  auto fooMeta = Struct_(Foo, StructMember_(Foo, a, &floatType));

  Fixture fixture;
  size_t N = 10;
  size_t Offset = 5;
  auto as = fixture.mallocLocal<float>(N);
  std::iota(as, as + N, 0);

  Foo expected{as + Offset};
  auto actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(std::memcmp(actual.a - Offset, as, sizeof(float) * N) == 0);
  CHECK(fixture.remoteAllocations.size() == 2);
}

TEST_CASE("ptr-offset-alias-shares-one-mirror") {
  struct Foo {
    float *a;
    float *b;
  };
  auto fooMeta = Struct_(Foo,                               //
                         StructMember_(Foo, a, &floatType), //
                         StructMember_(Foo, b, &floatType)  //
  );
  Fixture fixture;
  size_t N = 10;
  size_t Offset = 5;
  auto as = fixture.mallocLocal<float>(N);
  std::iota(as, as + N, 0);

  Foo expected{as + Offset, as};
  auto actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(actual.a == actual.b + Offset);
  CHECK(fixture.remoteAllocations.size() == 2);

  actual.a[0] = 77;
  actual.b[1] = 88;
  CHECK(actual.b[Offset] == 77);
  const size_t reads = fixture.remoteReads;
  fixture.allocation.syncRemoteToLocalAfterLaunch(&expected);
  CHECK(fixture.remoteReads - reads == 2);
  CHECK(as[Offset] == 77);
  CHECK(as[1] == 88);
  CHECK(expected.a == as + Offset);
  CHECK(expected.b == as);
}

TEST_CASE("ptr-offset-into-pointer-bearing-array") {
  struct Rec {
    int32_t tag;
    int32_t *data;
  };
  struct Foo {
    Rec *x;
    Rec *y;
  };
  auto recMeta = Struct_(Rec,                                 //
                         StructMember_(Rec, tag, &int32Type), //
                         StructMember_(Rec, data, &int32Type));
  auto fooMeta = Struct_(Foo,                              //
                         StructMember_(Foo, x, &*recMeta), //
                         StructMember_(Foo, y, &*recMeta));

  constexpr size_t N = 6, J = 3;
  const auto setup = [&](Fixture &fixture) {
    auto pool = fixture.mallocLocal<int32_t>(N);
    std::iota(pool, pool + N, 100);
    auto recs = fixture.mallocLocal<Rec>(N);
    for (size_t i = 0; i < N; ++i)
      recs[i] = Rec{static_cast<int32_t>(i), pool + i};
    return recs;
  };

  SECTION("only the interior pointer into an array types the whole array") {
    Fixture fixture;
    auto recs = setup(fixture);
    const Foo actual = fixture.localToRemote(Foo{recs + J, nullptr}, *fooMeta);
    CHECK(actual.x[-2].tag == 1);
    CHECK(actual.x[-1].tag == 2);
    CHECK(actual.x[-1].data != recs[2].data);
    CHECK(*actual.x[-1].data == 102);
    CHECK(*actual.x[0].data == 103);
    CHECK(*actual.x[1].data == 104);
    CHECK(fixture.remoteAllocations.size() == 3);
  }
  SECTION("a pointer to a member keeps the bytes before it raw") {
    struct Wrap {
      int32_t pad[2];
      Rec rec;
    };
    Fixture fixture;
    auto recs = setup(fixture);
    auto wrap = fixture.mallocLocal<Wrap>();
    wrap->pad[0] = 11;
    wrap->pad[1] = 12;
    wrap->rec = recs[1];
    const Foo actual = fixture.localToRemote(Foo{&wrap->rec, nullptr}, *fooMeta);
    CHECK(reinterpret_cast<const int32_t *>(actual.x)[-1] == 12);
    CHECK(reinterpret_cast<const int32_t *>(actual.x)[-2] == 11);
    CHECK(actual.x->tag == 1);
    CHECK(*actual.x->data == 101);
    CHECK(fixture.remoteAllocations.size() == 3);
  }
  SECTION("interior pointer then base: the earlier elements are typed too") {
    Fixture fixture;
    auto recs = setup(fixture);
    const Foo actual = fixture.localToRemote(Foo{recs + J, recs}, *fooMeta);
    CHECK(actual.x == actual.y + J);
    CHECK(actual.x[-1].data != recs[J - 1].data);
    CHECK(*actual.x[-1].data == 102);
    CHECK(*actual.y[0].data == 100);
    CHECK(*actual.x[0].data == 103);
    CHECK(fixture.remoteAllocations.size() == 3);
  }
  SECTION("base then interior pointer") {
    Fixture fixture;
    auto recs = setup(fixture);
    const Foo actual = fixture.localToRemote(Foo{recs, recs + J}, *fooMeta);
    CHECK(actual.y == actual.x + J);
    CHECK(actual.y[-1].data != recs[J - 1].data);
    CHECK(*actual.y[-1].data == 102);
    CHECK(fixture.remoteAllocations.size() == 3);
  }
}

TEST_CASE("an end pointer in the trailing slack of an allocation keeps the mirror of the elements before it") {
  struct Elem {
    int32_t tag;
    int32_t *data;
  };
  struct Span {
    Elem *begin;
    Elem *end;
  };
  auto elemMeta = Struct_(Elem,                                 //
                          StructMember_(Elem, tag, &int32Type), //
                          StructMember_(Elem, data, &int32Type));
  auto spanMeta = Struct_(Span,                                   //
                          StructMember_(Span, begin, &*elemMeta), //
                          StructMember_(Span, end, &*elemMeta));

  constexpr size_t N = 4, Head = 32, Slack = 7;
  const auto run = [&](const bool endFirst) {
    Fixture fixture;
    auto pool = fixture.mallocLocal<int32_t>(N);
    std::iota(pool, pool + N, 100);
    auto block = fixture.mallocLocalHeap<char>(Head + N * sizeof(Elem) + Slack);
    std::memset(block, 0, Head + N * sizeof(Elem) + Slack);
    auto elems = reinterpret_cast<Elem *>(block + Head);
    for (size_t i = 0; i < N; ++i)
      elems[i] = Elem{static_cast<int32_t>(i), pool + i};

    const Span actual =
        endFirst ? fixture.localToRemote(Span{elems + N, elems}, *spanMeta) : fixture.localToRemote(Span{elems, elems + N}, *spanMeta);
    Elem *begin = endFirst ? actual.end : actual.begin;
    Elem *end = endFirst ? actual.begin : actual.end;
    REQUIRE(fixture.remoteLive(begin));
    CHECK(fixture.releasedRemote.empty());
    CHECK(fixture.remoteAllocations.size() == 3);
    CHECK(end == begin + N);
    for (size_t i = 0; i < N; ++i) {
      CHECK(begin[i].tag == static_cast<int32_t>(i));
      CHECK(begin[i].data != pool + i);
      CHECK(*begin[i].data == static_cast<int32_t>(100 + i));
    }
  };
  SECTION("begin then end") { run(false); }
  SECTION("end then begin") { run(true); }
}

TEST_CASE("a record smaller than its pointee is mirrored at the pointee size") {
  struct Wide {
    int64_t value;
  };
  struct Capture {
    Wide *wide;
  };
  auto wideMeta = Struct_(Wide, StructMember_(Wide, value, &int64Type));
  auto captureMeta = Struct_(Capture, StructMember_(Capture, wide, &*wideMeta));

  Fixture fixture;
  auto *slot = static_cast<int32_t *>(std::malloc(sizeof(Wide)));
  fixture.localAllocations.emplace(reinterpret_cast<uintptr_t>(slot), sizeof(int32_t));
  *slot = 7;
  const Capture actual = fixture.localToRemote(Capture{reinterpret_cast<Wide *>(slot)}, *captureMeta);
  CHECK(fixture.remoteLive(actual.wide));
  CHECK(std::any_of(fixture.remoteAllocations.begin(), fixture.remoteAllocations.end(), [&](const auto &entry) {
    return entry.first == reinterpret_cast<uintptr_t>(actual.wide) && entry.second >= sizeof(Wide);
  }));
}

TEST_CASE("aliases of an undersized record with different pointee widths share one mirror") {
  struct Wide {
    int64_t lane[8];
  };
  struct NarrowFirst {
    int32_t *narrow;
    Wide *wide;
  };
  struct WideFirst {
    Wide *wide;
    int32_t *narrow;
  };
  auto wideMeta = Struct_(Wide, StructMember_(Wide, lane, &int64Type));
  auto narrowFirstMeta = Struct_(NarrowFirst,                                    //
                                 StructMember_(NarrowFirst, narrow, &int32Type), //
                                 StructMember_(NarrowFirst, wide, &*wideMeta));
  auto wideFirstMeta = Struct_(WideFirst,                                  //
                               StructMember_(WideFirst, wide, &*wideMeta), //
                               StructMember_(WideFirst, narrow, &int32Type));

  const auto run = [&](const auto &meta, const auto &capture, Fixture &fixture, const auto *slot) {
    const auto actual = fixture.localToRemote(capture, *meta);
    REQUIRE(fixture.remoteLive(actual.narrow));
    REQUIRE(fixture.remoteLive(actual.wide));
    CHECK(static_cast<const void *>(actual.narrow) == static_cast<const void *>(actual.wide));
    CHECK(std::memcmp(actual.wide, slot, sizeof(Wide)) == 0);
    CHECK(*actual.narrow == *reinterpret_cast<const int32_t *>(slot));
  };
  const auto slotOf = [](Fixture &fixture) {
    auto slot = fixture.mallocLocal<Wide>();
    for (size_t i = 0; i < 8; ++i)
      slot->lane[i] = 0x0101010101010101LL * static_cast<int64_t>(i + 1);
    fixture.localAllocations[reinterpret_cast<uintptr_t>(slot)] = sizeof(int32_t);
    return slot;
  };

  SECTION("narrow first") {
    Fixture fixture;
    auto slot = slotOf(fixture);
    run(narrowFirstMeta, NarrowFirst{reinterpret_cast<int32_t *>(slot), slot}, fixture, slot);
  }
  SECTION("wide first") {
    Fixture fixture;
    auto slot = slotOf(fixture);
    run(wideFirstMeta, WideFirst{slot, reinterpret_cast<int32_t *>(slot)}, fixture, slot);
  }
}

TEST_CASE("a generated mirror grown after a pointer was patched keeps that pointer live") {
  struct Capture {
    int32_t *narrow;
    char *wide;
  };
  constexpr size_t WideBytes = 64;
  Fixture fixture;
  auto slot = fixture.mallocLocal<char>(WideBytes);
  for (size_t i = 0; i < WideBytes; ++i)
    slot[i] = static_cast<char>(i + 1);
  fixture.localAllocations[reinterpret_cast<uintptr_t>(slot)] = sizeof(int32_t);

  Capture capture{reinterpret_cast<int32_t *>(slot), slot};
  const uintptr_t remote = fixture.allocation.mirrorAlloc(&capture, sizeof(capture), false);
  const uintptr_t narrow = fixture.allocation.mirrorEnsure(slot, sizeof(int32_t), true);
  fixture.allocation.mirrorPatch(remote, offsetof(Capture, narrow), narrow);
  const uintptr_t wide = fixture.allocation.mirrorEnsure(slot, WideBytes, true);
  fixture.allocation.mirrorPatch(remote, offsetof(Capture, wide), wide);

  const auto *actual = reinterpret_cast<const Capture *>(remote);
  REQUIRE(fixture.remoteLive(actual->narrow));
  REQUIRE(fixture.remoteLive(actual->wide));
  CHECK(static_cast<const void *>(actual->narrow) == static_cast<const void *>(actual->wide));
  CHECK(std::memcmp(actual->wide, slot, WideBytes) == 0);

  fixture.allocation.mirrorPatch(narrow, sizeof(uintptr_t), remote);
  CHECK(reinterpret_cast<const uintptr_t *>(actual->wide)[1] == remote);
}

TEST_CASE("resident arena leaves keep kernel output on the device until the host touches it") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  Fixture f;
  struct Rec {
    int32_t *xs;
    uint64_t n;
  };
  auto recLayout = Struct_(Rec, StructMember_(Rec, xs, &int32Type), StructMember_(Rec, n, &int64Type));
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  auto xs = f.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  auto rec = f.mallocLocal<Rec>(1);
  rec->xs = xs, rec->n = count;

  const auto launch = [&](const auto &kernel) {
    f.allocation.genArenaReset();
    f.allocation.genArenaMirror(reinterpret_cast<const char *>(rec), 1, 1, *recLayout, sizeof(uintptr_t), sizeof(Rec));
    auto *base = reinterpret_cast<char *>(f.allocation.genArenaFinish());
    const uint64_t off = *reinterpret_cast<const uint64_t *>(base + offsetof(Rec, xs));
    kernel(reinterpret_cast<int32_t *>(base + off));
    f.allocation.genArenaReadback();
  };

  launch([&](int32_t *d) { std::fill(d, d + count, 2); });
  const auto reads = f.remoteReads;
  CHECK(xs[0] == 2);
  CHECK(xs[count / 2] == 2);
  CHECK(f.remoteReads == reads + 1);

  launch([&](int32_t *d) {
    CHECK(d[count / 2] == 2);
    d[count / 2] = 3;
  });
  CHECK(xs[count / 2] == 3);

  xs[count / 2 + 1] = 7;
  launch([&](int32_t *d) { CHECK(d[count / 2 + 1] == 7); });

  f.allocation.forget(xs, count * sizeof(int32_t), false);
  xs[count / 2] = 9;
  CHECK(xs[count / 2] == 9);
}

TEST_CASE("gen-arena-mirror") {
  Fixture f;
  f.allocation.genArenaPersist = false;
  struct Rec {
    int8_t *p;
    uint64_t len;
  };
  auto i8 = NamedType(int8_t);
  auto recLayout = Struct_(Rec, StructMember_(Rec, p, &i8), StructMember_(Rec, len, &int64Type));

  auto *chars = f.mallocLocal<int8_t>(4);
  chars[0] = 'a', chars[1] = 'b', chars[2] = 'c', chars[3] = 0;
  auto *rec = f.mallocLocal<Rec>(1);
  rec->p = chars, rec->len = 3;

  // the whole reachable graph laid into one arena: Rec body first, then its char payload appended,
  // the `p` slot rewritten to the chars' arena byte offset
  const uint64_t off = f.allocation.genArenaMirror(reinterpret_cast<const char *>(rec), 1, 1, *recLayout, sizeof(uintptr_t), sizeof(Rec));
  auto &arena = f.allocation.genArenaStaging;
  CHECK(off == 0);
  CHECK(arena.size() == sizeof(Rec) + 4);
  CHECK(*reinterpret_cast<const uint64_t *>(arena.data() + offsetof(Rec, p)) == sizeof(Rec));
  CHECK(*reinterpret_cast<const uint64_t *>(arena.data() + offsetof(Rec, len)) == 3);
  CHECK(std::memcmp(arena.data() + sizeof(Rec), "abc", 3) == 0);

  // a null pointer slot becomes the arenaNull sentinel (0); the capture root sits at offset 0 with
  // nothing pointing back at it, so every real pointee offset is > 0 and 0 unambiguously means null
  f.allocation.genArenaReset();
  rec->p = nullptr;
  f.allocation.genArenaMirror(reinterpret_cast<const char *>(rec), 1, 1, *recLayout, sizeof(uintptr_t), sizeof(Rec));
  CHECK(*reinterpret_cast<const uint64_t *>(f.allocation.genArenaStaging.data() + offsetof(Rec, p)) == 0);
}

TEST_CASE("an arena pointee starting inside its allocation copies only the host bytes") {
  struct Wide {
    int64_t value;
  };
  struct Capture {
    Wide *wide;
  };
  auto wideMeta = Struct_(Wide, StructMember_(Wide, value, &int64Type));
  auto captureMeta = Struct_(Capture, StructMember_(Capture, wide, &*wideMeta));

  Fixture f;
  f.allocation.genArenaPersist = false;
  auto *block = f.mallocLocal<char>(8);
  std::memset(block, 0xab, 8);
  auto *capture = f.mallocLocal<Capture>(1);
  capture->wide = reinterpret_cast<Wide *>(block + 4);
  f.allocation.genArenaMirror(reinterpret_cast<const char *>(capture), 1, 1, *captureMeta, sizeof(uintptr_t), sizeof(Capture));
  const auto &staging = f.allocation.genArenaStaging;
  const size_t blockAt = (sizeof(Capture) + alignof(Wide) - 1) & ~(alignof(Wide) - 1);
  CHECK(staging.size() == blockAt + 12);
  CHECK(*reinterpret_cast<const uintptr_t *>(staging.data() + offsetof(Capture, wide)) == blockAt + 4);
  CHECK(std::memcmp(staging.data() + blockAt, block, 8) == 0);
  CHECK(std::all_of(staging.begin() + blockAt + 8, staging.end(), [](const char c) { return c == 0; }));
}

TEST_CASE("a straddling arena pointee reads back only the host bytes") {
  struct Wide {
    int64_t value;
  };
  struct Capture {
    Wide *wide;
  };
  auto wideMeta = Struct_(Wide, StructMember_(Wide, value, &int64Type));
  auto captureMeta = Struct_(Capture, StructMember_(Capture, wide, &*wideMeta));

  Fixture f;
  f.allocation.genArenaPersist = false;
  auto *block = f.mallocLocal<char>(8);
  auto *capture = f.mallocLocal<Capture>(1);
  capture->wide = reinterpret_cast<Wide *>(block + 4);

  f.allocation.genArenaReset();
  f.allocation.genArenaMirror(reinterpret_cast<const char *>(capture), 1, 1, *captureMeta, sizeof(uintptr_t), sizeof(Capture));
  auto *base = reinterpret_cast<char *>(f.allocation.genArenaFinish());
  auto *wide = reinterpret_cast<Wide *>(base + *reinterpret_cast<const uint64_t *>(base + offsetof(Capture, wide)));
  wide->value = 42;
  f.allocation.genArenaReadback();
  CHECK(*reinterpret_cast<const int32_t *>(block + 4) == 42);
}

TEST_CASE("a capture that references package memory is rejected when the device has no heap") {
  struct Grid {
    int32_t *begin;
    int32_t *end;
  };
  auto gridLayout = Struct_(Grid, StructMember_(Grid, begin, &int32Type), StructMember_(Grid, end, &int32Type));
  WithStubDevice device(false, false, 1024, 1, ModuleFormat::SPIRV_GLCompute);
  WithStubManagedPlatform platform;
  device.stub->packAllocations = true;
  device.stub->advertised = std::vector<std::string>{"overread_pad:4096"};
  StubQueue queue(std::chrono::seconds(1));
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  Fixture fixture;
  polyregion::polyrt::sma::CaptureMarshalOf<decltype(fixture.allocation)> marshal{fixture.allocation};
  polyregion::polyrt::sma::registerCaptureMarshal(&marshal);
  struct Unregister {
    ~Unregister() { polyregion::polyrt::sma::registerCaptureMarshal(nullptr); }
  } unregister;

  const auto begin = remoteAlloc(&context, 100 * sizeof(int32_t));
  Grid grid{reinterpret_cast<int32_t *>(begin), reinterpret_cast<int32_t *>(begin + 100 * sizeof(int32_t))};

  const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::Ptr)};
  const std::array<void *, 1> arguments{&grid};
  const std::array<size_t, 1> mirrorSizes{sizeof(Grid)};
  const std::array<uint8_t, 1> mirrorKinds{1};
  const std::array<const TypeLayout *, 1> layouts{&*gridLayout};
  CHECK_FALSE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                              arguments.data(), mirrorSizes.data(), mirrorKinds.data(), layouts.data()));
  CHECK(std::string(polyrt_error_message()).find("package memory") != std::string::npos);
  CHECK(queue.firstPointer == 0);
}

TEST_CASE("a stdpar capture that references package memory is rejected when the device has no heap") {
  struct Grid {
    int32_t *begin;
    int32_t *end;
  };
  auto gridLayout = Struct_(Grid, StructMember_(Grid, begin, &int32Type), StructMember_(Grid, end, &int32Type));
  WithStubDevice device(false, false, 1024, 1, ModuleFormat::SPIRV_GLCompute);
  WithStubManagedPlatform platform;
  device.stub->packAllocations = true;
  StubQueue queue(std::chrono::seconds(1));
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  Fixture fixture;

  const auto begin = remoteAlloc(&context, 100 * sizeof(int32_t));
  Grid grid{reinterpret_cast<int32_t *>(begin), reinterpret_cast<int32_t *>(begin + 100 * sizeof(int32_t))};
  CHECK_THROWS_AS(
      polyregion::polyrt::sma::arenaMarshal(fixture.allocation, &grid, *gridLayout, polyregion::polyrt::sma::arenaPlacementOf(&context)),
      std::logic_error);

  auto *host = fixture.mallocLocal<int32_t>(16);
  Grid plain{host, host + 16};
  CHECK_NOTHROW(
      polyregion::polyrt::sma::arenaMarshal(fixture.allocation, &plain, *gridLayout, polyregion::polyrt::sma::arenaPlacementOf(&context)));
}

namespace {
struct RootedMarshal final : polyregion::polyrt::sma::CaptureMarshal {
  polyregion::polyrt::sma::CaptureMarshal &inner;
  int64_t root;
  RootedMarshal(polyregion::polyrt::sma::CaptureMarshal &inner, const int64_t root) : inner(inner), root(root) {}
  void visitClear() override { inner.visitClear(); }
  uintptr_t ensure(const void *local, const size_t minSize) override { return inner.ensure(local, minSize); }
  void readAlloc(const void *local) override { inner.readAlloc(local); }
  uintptr_t mirror(const void *capture, const TypeLayout &layout) override { return inner.mirror(capture, layout); }
  void readMirror(void *capture) override { inner.readMirror(capture); }
  polyregion::polyrt::sma::ArenaImage arena(const void *capture, const TypeLayout &layout,
                                            polyregion::polyrt::ArenaPlacement placement) override {
    auto image = inner.arena(capture, layout, std::move(placement));
    image.root = root;
    return image;
  }
  void readArena() override { inner.readArena(); }
  void releaseArena() override { inner.releaseArena(); }
};

struct RootLaunch {
  StubQueue queue{std::chrono::seconds(1)};
  WithStubDevice device{false, false, 1024, 1, ModuleFormat::SPIRV_GLCompute};
  WithStubManagedPlatform platform;
  polyregion::polyrt::ExecutionContext context{polyregion::polyrt::currentPlatform.get(), device.stub, &queue};
  Fixture fixture;
  polyregion::polyrt::sma::CaptureMarshalOf<decltype(fixture.allocation)> plain{fixture.allocation};
  RootedMarshal rooted;
  struct Span {
    int32_t *data;
    int64_t length;
  } span{};

  explicit RootLaunch(const int64_t root, const std::optional<size_t> viewStart) : rooted(plain, root) {
    span = {fixture.mallocLocal<int32_t>(4), 4};
    device.stub->advertised = std::vector<std::string>{"overread_pad:4096"};
    device.stub->viewStart = viewStart;
    polyregion::polyrt::sma::registerCaptureMarshal(&rooted);
  }
  ~RootLaunch() { polyregion::polyrt::sma::registerCaptureMarshal(nullptr); }

  void launch() {
    auto layout = Struct_(Span, StructMember_(Span, data, &int32Type), StructMember_(Span, length, &int64Type));
    const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::Ptr)};
    const std::array<void *, 1> arguments{&span};
    const std::array<size_t, 1> mirrorSizes{sizeof(Span)};
    const std::array<uint8_t, 1> mirrorKinds{1};
    const std::array<const TypeLayout *, 1> layouts{&*layout};
    REQUIRE(polyrt_remote_launch_marshalled(&context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                            arguments.data(), mirrorSizes.data(), mirrorKinds.data(), layouts.data()));
  }

  template <typename T> T argument(const size_t index) {
    const auto pointers = detail::argDataAsPointers(queue.invokedTypes, queue.invokedData);
    T value{};
    std::memcpy(&value, pointers[index], sizeof(T));
    return value;
  }
};
} // namespace

TEST_CASE("a logical arena launch passes the arena offset of the capture ahead of the views") {
  for (const int64_t root : {int64_t{0}, int64_t{4096}, int64_t{1} << 40}) {
    RootLaunch f(root, 0);
    f.launch();
    REQUIRE(f.queue.invokedTypes.size() == 1 + polyregion::polyrt::sma::arenaViewCount + 1);
    CHECK(f.queue.invokedTypes.front() == Type::IntS64);
    CHECK(f.argument<int64_t>(0) == root);
    for (size_t view = 1; view <= polyregion::polyrt::sma::arenaViewCount; ++view) {
      CHECK(f.queue.invokedTypes[view] == Type::Ptr);
      CHECK(f.argument<uintptr_t>(view) == f.argument<uintptr_t>(1));
    }
  }
}

TEST_CASE("a launch without arena views gets no root argument") {
  RootLaunch f(4096, std::nullopt);
  f.launch();
  CHECK(f.queue.invokedTypes == std::vector<Type>{Type::Ptr, Type::Void});
}

namespace {
Platform *configureHeapDevice(StubDevice *stub, const std::optional<HeapLimits> heap) {
  stub->heap = heap;
  stub->advertised = std::vector<std::string>{"overread_pad:4096"};
  return polyregion::polyrt::currentPlatform.get();
}

struct HeapRouting {
  WithStubDevice device;
  StubQueue queue{std::chrono::seconds(1)};
  std::unique_ptr<polyregion::polyrt::ExecutionContext> owned;
  polyregion::polyrt::ExecutionContext &context;

  explicit HeapRouting(const std::optional<HeapLimits> heap, const ModuleFormat format = ModuleFormat::SPIRV_GLCompute)
      : device(false, false, 1024, 1, format),
        owned(std::make_unique<polyregion::polyrt::ExecutionContext>(configureHeapDevice(device.stub, heap), device.stub, &queue)),
        context(*owned) {}

  void retire() { owned.reset(); }

  PolyrtStats stats() {
    PolyrtStats stats{};
    REQUIRE(polyrt_stats(&context, &stats));
    return stats;
  }
};
} // namespace

TEST_CASE("glcompute package allocations come from one aligned device heap") {
  HeapRouting f(HeapLimits{size_t{1} << 24, 64});
  const auto a = remoteAlloc(&f.context, 100);
  const auto b = remoteAlloc(&f.context, 200);
  CHECK(f.device.stub->remoteAllocs == 0);
  CHECK(a % 64 == 0);
  CHECK(b % 64 == 0);
  CHECK(f.context.remoteAllocations.size() == 2);
  const auto *tail = reinterpret_cast<const uint8_t *>(a);
  for (size_t i = 100; i < 100 + 1 + 4096; ++i)
    REQUIRE(tail[i] == 0);
  CHECK(f.stats().remoteMallocs == 2);
  CHECK(f.stats().heapBytesLive > 300);
  CHECK(f.stats().heapBytesCommitted >= f.stats().heapBytesLive);

  REQUIRE(polyrt_remote_free(&f.context, a));
  REQUIRE(polyrt_remote_free(&f.context, b));
  CHECK(f.device.stub->remoteFrees == 0);
  CHECK(f.stats().heapBytesLive == 0);
  CHECK(f.context.remoteAllocations.empty());
}

TEST_CASE("glcompute temporaries return to the device heap when the scope releases") {
  HeapRouting f(HeapLimits{size_t{1} << 24, 64});
  REQUIRE(polyrt_context_acquire(&f.context));
  const auto temporary = remoteTempAlloc(&f.context, 1000);
  CHECK(temporary != 0);
  CHECK(f.stats().heapBytesLive >= 1000);
  REQUIRE(polyrt_context_release(&f.context));
  CHECK(f.stats().heapBytesLive == 0);
  CHECK(f.context.remoteAllocations.empty());
}

TEST_CASE("only glcompute devices with heap limits use the device heap") {
  {
    HeapRouting f(HeapLimits{size_t{1} << 24, 64}, ModuleFormat::Object);
    const auto pointer = remoteAlloc(&f.context, 64);
    CHECK(f.device.stub->remoteAllocs == 1);
    CHECK(f.stats().heapBytesCommitted == 0);
    REQUIRE(polyrt_remote_free(&f.context, pointer));
    CHECK(f.device.stub->remoteFrees == 1);
  }
  {
    HeapRouting f(std::nullopt);
    const auto pointer = remoteAlloc(&f.context, 64);
    CHECK(f.device.stub->remoteAllocs == 1);
    REQUIRE(polyrt_remote_free(&f.context, pointer));
    CHECK(f.device.stub->remoteFrees == 1);
  }
}

TEST_CASE("device heap exhaustion is a status failure that names the limit") {
  HeapRouting f(HeapLimits{size_t{1} << 16, 64});
  uintptr_t result = 1;
  CHECK_FALSE(polyrt_remote_malloc(&f.context, size_t{1} << 17, &result));
  CHECK(result == 0);
  const std::string message = polyrt_error_message();
  CHECK(message.find("exhausted") != std::string::npos);
  CHECK(message.find("65536") != std::string::npos);
  CHECK(f.context.remoteAllocations.empty());
  CHECK(polyrt_remote_malloc(&f.context, 100, &result));
  CHECK(result != 0);
}

namespace {
struct HeapArena {
  WithStubManagedPlatform platform;
  HeapRouting f;
  Fixture fixture;
  polyregion::polyrt::sma::CaptureMarshalOf<decltype(fixture.allocation)> marshal{fixture.allocation};

  explicit HeapArena(const size_t limit = size_t{1} << 26, const ModuleFormat format = ModuleFormat::SPIRV_GLCompute)
      : f(HeapLimits{limit, 64}, format) {
    f.device.stub->viewStart = 0;
    polyregion::polyrt::sma::registerCaptureMarshal(&marshal);
  }
  ~HeapArena() { polyregion::polyrt::sma::registerCaptureMarshal(nullptr); }

  uintptr_t heap() { return fixture.allocation.genArenaDevBase; }
  uint64_t window() { return fixture.allocation.genArenaWindow; }
  const char *image(const uint64_t offset = 0) { return reinterpret_cast<const char *>(heap() + window() + offset); }
  const char *at(const uint64_t heapOffset) { return reinterpret_cast<const char *>(heap() + heapOffset); }

  bool launch(void *capture, const TypeLayout &layout) {
    const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::Ptr)};
    const std::array<void *, 1> arguments{capture};
    const std::array<size_t, 1> mirrorSizes{layout.sizeInBytes};
    const std::array<uint8_t, 1> mirrorKinds{1};
    const std::array<const TypeLayout *, 1> layouts{&layout};
    return polyrt_remote_launch_marshalled(&f.context, "module", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                           arguments.data(), mirrorSizes.data(), mirrorKinds.data(), layouts.data());
  }

  template <typename T> T argument(const size_t index) {
    const auto pointers = detail::argDataAsPointers(f.queue.invokedTypes, f.queue.invokedData);
    T value{};
    std::memcpy(&value, pointers[index], sizeof(T));
    return value;
  }

  uintptr_t slot(const size_t offset) {
    uintptr_t value{};
    std::memcpy(&value, image(offset), sizeof(value));
    return value;
  }

  void upload(const uintptr_t block, const void *host, const size_t bytes) {
    REQUIRE(polyrt_remote_memcpy(&f.context, block, reinterpret_cast<uintptr_t>(host), bytes, 0));
  }
};

struct Grid {
  int32_t *first;
  int32_t *second;
  int32_t *host;
};
} // namespace

TEST_CASE("heap devices keep the assert buffer in one device heap block") {
  HeapArena a;
  const unsigned char image = 0;
  std::array<const char *, 1> asserting{polyregion::conventions::ErrorArg};
  REQUIRE(polyrt_remote_load(&a.f.context, "asserting_heap", static_cast<int32_t>(PlatformKind::Managed),
                             static_cast<int32_t>(ModuleFormat::SPIRV_GLCompute), asserting.size(), asserting.data(), sizeof(image),
                             &image));
  int64_t n = 3;
  const std::array<uint8_t, 1> types{static_cast<uint8_t>(Type::IntS64)};
  const std::array<void *, 1> arguments{&n};
  for (size_t launch = 0; launch < 6; ++launch)
    REQUIRE(polyrt_remote_launch(&a.f.context, "asserting_heap", "kernel", 1, 1, 1, 1, 1, 1, 0, arguments.size(), types.data(),
                                 arguments.data()));
  CHECK(a.f.stats().remoteMallocs == 1);
  CHECK(a.f.stats().heapBytesLive >= polyregion::polyrt::assertBufferBytes);
  CHECK(a.f.device.stub->remoteAllocs == 0);
}

TEST_CASE("arena binds a one-past-end remote pointer that lives in the device heap") {
  struct Span {
    int32_t *begin;
    int32_t *end;
  };
  auto spanLayout = Struct_(Span, StructMember_(Span, begin, &int32Type), StructMember_(Span, end, &int32Type));
  HeapArena a;
  const auto begin = remoteAlloc(&a.f.context, 100 * sizeof(int32_t));
  remoteAlloc(&a.f.context, 100 * sizeof(int32_t));
  Span span{reinterpret_cast<int32_t *>(begin), reinterpret_cast<int32_t *>(begin + 100 * sizeof(int32_t))};
  REQUIRE(a.launch(&span, *spanLayout));
  CHECK(a.slot(offsetof(Span, begin)) == begin - a.heap());
  CHECK(a.slot(offsetof(Span, end)) - a.slot(offsetof(Span, begin)) == 100 * sizeof(int32_t));
}

TEST_CASE("arena launches with two bound heap blocks copy no device bytes and wait once") {
  auto gridLayout =
      Struct_(Grid, StructMember_(Grid, first, &int32Type), StructMember_(Grid, second, &int32Type), StructMember_(Grid, host, &int32Type));
  HeapArena a;
  const auto first = remoteAlloc(&a.f.context, 100 * sizeof(int32_t));
  const auto second = remoteAlloc(&a.f.context, 50 * sizeof(int32_t));
  auto *host = a.fixture.mallocLocal<int32_t>(16);
  std::iota(host, host + 16, 7);
  Grid grid{reinterpret_cast<int32_t *>(first), reinterpret_cast<int32_t *>(second), host};

  REQUIRE(a.launch(&grid, *gridLayout));
  constexpr uint64_t launches = 100;
  const auto before = a.f.stats();
  for (uint64_t i = 0; i < launches; ++i) {
    REQUIRE(a.launch(&grid, *gridLayout));
    CHECK(a.slot(offsetof(Grid, first)) == first - a.heap());
    CHECK(a.slot(offsetof(Grid, second)) == second - a.heap());
    CHECK(std::memcmp(a.at(a.slot(offsetof(Grid, host))), host, 16 * sizeof(int32_t)) == 0);
  }
  const auto after = a.f.stats();
  INFO("waits per launch: " << static_cast<double>(after.waits - before.waits) / launches);
  CHECK(after.arenaLaunches - before.arenaLaunches == launches);
  CHECK(after.d2dCopies - before.d2dCopies == 0);
  CHECK(after.waits - before.waits <= launches);
}

TEST_CASE("growing the heap and the window keeps package handles and bytes") {
  auto gridLayout =
      Struct_(Grid, StructMember_(Grid, first, &int32Type), StructMember_(Grid, second, &int32Type), StructMember_(Grid, host, &int32Type));
  HeapArena a;
  constexpr size_t count = 100;
  std::vector<int32_t> pattern(count);
  std::iota(pattern.begin(), pattern.end(), 1000);
  const auto first = remoteAlloc(&a.f.context, count * sizeof(int32_t));
  const auto second = remoteAlloc(&a.f.context, count * sizeof(int32_t));
  a.upload(first, pattern.data(), pattern.size() * sizeof(int32_t));
  a.upload(second, pattern.data(), pattern.size() * sizeof(int32_t));
  auto *small = a.fixture.mallocLocal<int32_t>(16);
  std::iota(small, small + 16, 1);
  Grid grid{reinterpret_cast<int32_t *>(first), reinterpret_cast<int32_t *>(second), small};
  REQUIRE(a.launch(&grid, *gridLayout));

  const auto heap = a.heap();
  const auto firstSlot = a.slot(offsetof(Grid, first)), secondSlot = a.slot(offsetof(Grid, second));
  const auto before = a.f.stats();

  for (size_t i = 0; i < 64; ++i)
    remoteAlloc(&a.f.context, 1 << 14);
  const auto grown = a.f.stats();
  REQUIRE(grown.heapGrowths > before.heapGrowths);

  constexpr size_t bigCount = size_t{1} << 18;
  auto *big = a.fixture.mallocLocal<int32_t>(bigCount);
  std::iota(big, big + bigCount, 5);
  grid.host = big;
  REQUIRE(a.launch(&grid, *gridLayout));

  CHECK(a.heap() == heap);
  CHECK(a.slot(offsetof(Grid, first)) == firstSlot);
  CHECK(a.slot(offsetof(Grid, second)) == secondSlot);
  CHECK(std::memcmp(a.at(firstSlot), pattern.data(), pattern.size() * sizeof(int32_t)) == 0);
  CHECK(std::memcmp(a.at(secondSlot), pattern.data(), pattern.size() * sizeof(int32_t)) == 0);
  CHECK(std::memcmp(a.at(a.slot(offsetof(Grid, host))), big, bigCount * sizeof(int32_t)) == 0);
  CHECK(a.f.stats().arenaWindowMoves == 1);
  CHECK(a.f.stats().d2dCopies == 1);
}

TEST_CASE("a window that cannot grow in place moves with its resident leaf and guard fault reads follow it") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  struct Rec {
    int32_t *xs;
    uint64_t n;
    int32_t *extra;
  };
  auto recLayout =
      Struct_(Rec, StructMember_(Rec, xs, &int32Type), StructMember_(Rec, n, &int64Type), StructMember_(Rec, extra, &int32Type));
  HeapArena a;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  auto *xs = a.fixture.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  auto *small = a.fixture.mallocLocal<int32_t>(16);
  std::iota(small, small + 16, 1);
  Rec rec{xs, count, small};

  const auto leaf = [&] { return reinterpret_cast<int32_t *>(const_cast<char *>(a.at(a.slot(offsetof(Rec, xs))))); };

  a.f.queue.onInvoke = [&] { std::fill(leaf(), leaf() + count, 2); };
  REQUIRE(a.launch(&rec, *recLayout));
  const auto firstWindow = a.window();
  CHECK(a.f.stats().arenaWindowMoves == 0);

  const auto blocker = remoteAlloc(&a.f.context, 64);
  CHECK(blocker > a.heap() + firstWindow);
  xs[0] = 7;
  xs[count - 1] = 9;

  constexpr size_t bigCount = size_t{1} << 18;
  auto *big = a.fixture.mallocLocal<int32_t>(bigCount);
  std::iota(big, big + bigCount, 5);
  rec.extra = big;
  bool leafIntact = false;
  a.f.queue.onInvoke = [&] {
    leafIntact = leaf()[0] == 7 && leaf()[count / 2] == 2 && leaf()[count - 1] == 9;
    leaf()[count / 2] = 3;
  };
  REQUIRE(a.launch(&rec, *recLayout));

  CHECK(a.window() != firstWindow);
  CHECK(a.f.stats().arenaWindowMoves == 1);
  CHECK(a.f.stats().d2dCopies == 1);
  CHECK(leafIntact);
  CHECK(std::memcmp(a.at(a.slot(offsetof(Rec, extra))), big, bigCount * sizeof(int32_t)) == 0);
  CHECK(xs[0] == 7);
  CHECK(xs[count / 2] == 3);
  CHECK(xs[count - 1] == 9);
  a.fixture.allocation.forget(xs, count * sizeof(int32_t), false);
}

TEST_CASE("a union of pointers is rebased by the window offset once") {
  struct Either {
    union {
      int32_t *a;
      float *b;
    };
  };
  auto f32 = NamedType(float);
  auto layout = Struct_(Either, StructMember_(Either, a, &int32Type), StructMember_(Either, b, &f32));
  HeapArena a;
  auto *host = a.fixture.mallocLocal<int32_t>(16);
  std::iota(host, host + 16, 3);
  Either either{};
  either.a = host;
  REQUIRE(a.launch(&either, *layout));
  const auto value = a.slot(0);
  CHECK(value >= a.window());
  CHECK(value - a.window() < 4 * a.fixture.allocation.genArenaDevCapacity);
  CHECK(std::memcmp(a.at(value), host, 16 * sizeof(int32_t)) == 0);
}

TEST_CASE("read-back translates package, interior and cleared pointers through a non-zero window") {
  struct Rec {
    int32_t *package;
    int32_t *interior;
    int32_t *cleared;
    int32_t *other;
  };
  auto layout = Struct_(Rec, StructMember_(Rec, package, &int32Type), StructMember_(Rec, interior, &int32Type),
                        StructMember_(Rec, cleared, &int32Type), StructMember_(Rec, other, &int32Type));
  HeapArena a;
  const auto first = remoteAlloc(&a.f.context, 100 * sizeof(int32_t));
  const auto second = remoteAlloc(&a.f.context, 100 * sizeof(int32_t));
  auto *hostB = a.fixture.mallocLocal<int32_t>(16);
  auto *hostC = a.fixture.mallocLocal<int32_t>(16);
  std::iota(hostB, hostB + 16, 100);
  std::iota(hostC, hostC + 16, 200);
  Rec rec{reinterpret_cast<int32_t *>(first), hostB, hostC, hostC};
  a.f.queue.onInvoke = [&] {
    CHECK(a.window() != 0);
    const uintptr_t otherSlot = a.slot(offsetof(Rec, other));
    const uintptr_t package = second + 8 - a.heap(), interior = otherSlot + 4, cleared = 0;
    auto *image = const_cast<char *>(a.image());
    std::memcpy(image + offsetof(Rec, package), &package, sizeof(package));
    std::memcpy(image + offsetof(Rec, interior), &interior, sizeof(interior));
    std::memcpy(image + offsetof(Rec, cleared), &cleared, sizeof(cleared));
  };
  REQUIRE(a.launch(&rec, *layout));
  CHECK(reinterpret_cast<uintptr_t>(rec.package) == second + 8);
  CHECK(rec.interior == hostC + 1);
  CHECK(rec.cleared == nullptr);
  CHECK(rec.other == hostC);
}

TEST_CASE("an untouched resident leaf is carried across a window move") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  struct Rec {
    int32_t *xs;
    int32_t *extra;
  };
  auto layout = Struct_(Rec, StructMember_(Rec, xs, &int32Type), StructMember_(Rec, extra, &int32Type));
  HeapArena a;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  auto *xs = a.fixture.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  auto *small = a.fixture.mallocLocal<int32_t>(16);
  Rec rec{xs, small};
  a.f.queue.onInvoke = [&] {
    auto *leaf = reinterpret_cast<int32_t *>(const_cast<char *>(a.at(a.slot(offsetof(Rec, xs)))));
    std::iota(leaf, leaf + count, 10);
  };
  REQUIRE(a.launch(&rec, *layout));
  const auto window = a.window();
  remoteAlloc(&a.f.context, 64);

  constexpr size_t bigCount = size_t{1} << 18;
  auto *big = a.fixture.mallocLocal<int32_t>(bigCount);
  Rec other{nullptr, big};
  a.f.queue.onInvoke = nullptr;
  REQUIRE(a.launch(&other, *layout));
  CHECK(a.window() != window);
  CHECK(xs[count / 2] == static_cast<int32_t>(10 + count / 2));
  CHECK(xs[1] == 11);
  a.fixture.allocation.forget(xs, count * sizeof(int32_t), false);
}

TEST_CASE("a guard fault during a window move waits for the move and reads the moved bytes") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  struct Rec {
    int32_t *xs;
    int32_t *extra;
  };
  auto layout = Struct_(Rec, StructMember_(Rec, xs, &int32Type), StructMember_(Rec, extra, &int32Type));
  HeapArena a;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  auto *xs = a.fixture.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  auto *small = a.fixture.mallocLocal<int32_t>(16);
  Rec rec{xs, small};
  a.f.queue.onInvoke = [&] {
    auto *leaf = reinterpret_cast<int32_t *>(const_cast<char *>(a.at(a.slot(offsetof(Rec, xs)))));
    std::fill(leaf, leaf + count, 5);
  };
  REQUIRE(a.launch(&rec, *layout));
  a.f.queue.onInvoke = nullptr;
  remoteAlloc(&a.f.context, 64);

  constexpr size_t bigCount = size_t{1} << 18;
  auto *big = a.fixture.mallocLocal<int32_t>(bigCount);
  Rec grown{xs, big};
  std::atomic<bool> read{false};
  std::atomic<int32_t> seen{0};
  std::thread reader;
  bool readerWaited = false;
  a.f.queue.onCopy = [&] {
    reader = std::thread([&] {
      seen = xs[count / 2];
      read = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    readerWaited = !read;
    a.f.queue.onCopy = nullptr;
  };
  REQUIRE(a.launch(&grown, *layout));
  reader.join();
  CHECK(readerWaited);
  CHECK(seen == 5);
  a.fixture.allocation.forget(xs, count * sizeof(int32_t), false);
}

TEST_CASE("heap growth from another thread waits for an arena launch in flight") {
  auto gridLayout =
      Struct_(Grid, StructMember_(Grid, first, &int32Type), StructMember_(Grid, second, &int32Type), StructMember_(Grid, host, &int32Type));
  HeapArena a;
  auto *host = a.fixture.mallocLocal<int32_t>(16);
  Grid grid{nullptr, nullptr, host};
  REQUIRE(a.launch(&grid, *gridLayout));
  const auto growths = a.f.stats().heapGrowths;
  std::thread allocator;
  uintptr_t block = 0;
  bool grewDuringLaunch = true;
  a.f.queue.onInvoke = [&] {
    allocator = std::thread([&] { polyrt_remote_malloc(&a.f.context, size_t{1} << 23, &block); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    grewDuringLaunch = a.f.stats().heapGrowths != growths;
  };
  REQUIRE(a.launch(&grid, *gridLayout));
  allocator.join();
  CHECK_FALSE(grewDuringLaunch);
  CHECK(block != 0);
  CHECK(a.f.stats().heapGrowths > growths);
}

TEST_CASE("a device that is not glcompute keeps the arena at root zero even when it advertises heap limits") {
  struct Span {
    int32_t *begin;
    int32_t *end;
  };
  auto layout = Struct_(Span, StructMember_(Span, begin, &int32Type), StructMember_(Span, end, &int32Type));
  HeapArena a(size_t{1} << 26, ModuleFormat::SPIRV_Kernel);
  auto *host = a.fixture.mallocLocal<int32_t>(16);
  Span span{host, host + 16};
  REQUIRE(a.launch(&span, *layout));
  CHECK(a.argument<int64_t>(0) == 0);
  CHECK(a.window() == 0);
  CHECK(a.f.stats().heapBytesCommitted == 0);
}

TEST_CASE("a host pointer that falls in a free span of the device heap is mirrored, not taken for package memory") {
  struct Holder {
    int32_t *pointer;
  };
  auto layout = Struct_(Holder, StructMember_(Holder, pointer, &int32Type));
  HeapArena a;
  auto *host = a.fixture.mallocLocal<int32_t>(16);
  Holder holder{host};
  REQUIRE(a.launch(&holder, *layout));
  const auto committed = a.f.stats().heapBytesCommitted;
  const auto address = a.heap() + committed - 4096;
  a.fixture.localAllocations.emplace(address, 64);
  holder.pointer = reinterpret_cast<int32_t *>(address);
  REQUIRE(a.launch(&holder, *layout));
  const auto value = a.slot(0);
  CHECK(value != address - a.heap());
  CHECK(value >= a.window());
  a.fixture.localAllocations.erase(address);
}

TEST_CASE("destroying the context releases guarded arena leaves before the heap goes") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  struct Rec {
    int32_t *xs;
  };
  auto layout = Struct_(Rec, StructMember_(Rec, xs, &int32Type));
  HeapArena a;
  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  auto *xs = a.fixture.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  Rec rec{xs};
  a.f.queue.onInvoke = [&] {
    auto *leaf = reinterpret_cast<int32_t *>(const_cast<char *>(a.at(a.slot(offsetof(Rec, xs)))));
    std::fill(leaf, leaf + count, 4);
  };
  REQUIRE(a.launch(&rec, *layout));
  a.f.queue.onInvoke = nullptr;
  a.f.retire();
  CHECK(xs[count / 2] == 4);
  a.fixture.allocation.forget(xs, count * sizeof(int32_t), false);
}

TEST_CASE("a leaf uploaded into reused heap memory brings its zeroed over-read slack") {
  struct Span {
    int32_t *begin;
    int32_t *end;
  };
  auto layout = Struct_(Span, StructMember_(Span, begin, &int32Type), StructMember_(Span, end, &int32Type));
  HeapArena a;
  auto *host = a.fixture.mallocLocal<int32_t>(16);
  Span span{host, host + 16};
  REQUIRE(a.launch(&span, *layout));
  const auto committed = a.f.stats().heapBytesCommitted;
  std::memset(const_cast<char *>(a.image()), 0xFF, committed - a.window());
  REQUIRE(a.launch(&span, *layout));
  const auto leaf = a.slot(offsetof(Span, begin));
  const auto *after = a.at(leaf) + 16 * sizeof(int32_t);
  for (size_t i = 0; i < 4096; ++i)
    REQUIRE(after[i] == 0);
}

TEST_CASE("heap memcpy moves a multi-megabyte span in both directions and into a guarded leaf") {
  if (!polyregion::polyrt::guard::AtomicPublish) SKIP("no whole-page publish on this platform");
  struct Rec {
    int32_t *xs;
  };
  auto layout = Struct_(Rec, StructMember_(Rec, xs, &int32Type));
  HeapArena a;
  constexpr size_t bytes = size_t{10} << 20;
  std::vector<uint8_t> pattern(bytes), back(bytes, 0);
  std::iota(pattern.begin(), pattern.end(), 3);
  const auto block = remoteAlloc(&a.f.context, bytes);
  a.upload(block, pattern.data(), bytes);
  REQUIRE(polyrt_remote_memcpy(&a.f.context, reinterpret_cast<uintptr_t>(back.data()), block, bytes, 1));
  CHECK(back == pattern);
  const auto copy = remoteAlloc(&a.f.context, bytes);
  REQUIRE(polyrt_remote_memcpy(&a.f.context, copy, block, bytes, 2));
  std::fill(back.begin(), back.end(), 0);
  REQUIRE(polyrt_remote_memcpy(&a.f.context, reinterpret_cast<uintptr_t>(back.data()), copy, bytes, 1));
  CHECK(back == pattern);

  const size_t count = 64 * polyregion::polyrt::guard::pageSize() / sizeof(int32_t);
  auto *xs = a.fixture.mallocLocalHeap<int32_t>(count);
  std::fill(xs, xs + count, 1);
  Rec rec{xs};
  a.f.queue.onInvoke = [&] {
    auto *leaf = reinterpret_cast<int32_t *>(const_cast<char *>(a.at(a.slot(offsetof(Rec, xs)))));
    std::fill(leaf, leaf + count, 6);
  };
  REQUIRE(a.launch(&rec, *layout));
  REQUIRE(polyrt_remote_memcpy(&a.f.context, reinterpret_cast<uintptr_t>(xs), block, count * sizeof(int32_t), 1));
  CHECK(std::memcmp(xs, pattern.data(), count * sizeof(int32_t)) == 0);
  a.fixture.allocation.forget(xs, count * sizeof(int32_t), false);
}

TEST_CASE("a launch cannot move an arena between the device heap and a separate buffer") {
  auto gridLayout =
      Struct_(Grid, StructMember_(Grid, first, &int32Type), StructMember_(Grid, second, &int32Type), StructMember_(Grid, host, &int32Type));
  HeapArena a;
  auto *small = a.fixture.mallocLocal<int32_t>(16);
  Grid grid{nullptr, nullptr, small};
  REQUIRE(a.launch(&grid, *gridLayout));
  CHECK_THROWS_AS(polyregion::polyrt::sma::arenaMarshal(a.fixture.allocation, &grid, *gridLayout, {}), std::logic_error);
}

TEST_CASE("a window next to free space grows in place") {
  auto gridLayout =
      Struct_(Grid, StructMember_(Grid, first, &int32Type), StructMember_(Grid, second, &int32Type), StructMember_(Grid, host, &int32Type));
  HeapArena a;
  auto *small = a.fixture.mallocLocal<int32_t>(16);
  Grid grid{nullptr, nullptr, small};
  REQUIRE(a.launch(&grid, *gridLayout));
  const auto window = a.window();
  constexpr size_t bigCount = size_t{1} << 18;
  auto *big = a.fixture.mallocLocal<int32_t>(bigCount);
  std::iota(big, big + bigCount, 5);
  grid.host = big;
  REQUIRE(a.launch(&grid, *gridLayout));
  CHECK(a.window() == window);
  CHECK(a.f.stats().arenaWindowMoves == 0);
  CHECK(a.f.stats().d2dCopies == 0);
  CHECK(std::memcmp(a.at(a.slot(offsetof(Grid, host))), big, bigCount * sizeof(int32_t)) == 0);
}

TEST_CASE("stats are unavailable without a context") {
  PolyrtStats stats{};
  CHECK_FALSE(polyrt_stats(nullptr, &stats));
}

#ifdef POLYRT_TESTS_POLYDCO

TEST_CASE("polydco managed launches leave no device allocations behind") {
  WithStubDevice device(false, false);
  WithStubManagedPlatform platform;

  struct Capture {
    polyregion::polydco::FManagedPrelude bounds;
    int64_t scale;
    int64_t offset;
  };
  const auto boundsType = NamedType(polyregion::polydco::FManagedPrelude);
  auto layout = Struct_(Capture,                                     //
                        StructMember_(Capture, bounds, &boundsType), //
                        StructMember_(Capture, scale, &int64Type),   //
                        StructMember_(Capture, offset, &int64Type));

  const unsigned char image = 0;
  KernelObject object{};
  object.kind = PlatformKind::Managed;
  object.format = ModuleFormat::Object;
  object.imageLength = sizeof(image);
  object.image = &image;
  KernelBundle bundle{};
  bundle.moduleName = "polydco_capture_release";
  bundle.objectCount = 1;
  bundle.objects = &object;
  bundle.structCount = 1;
  bundle.structs = &*layout;

  const auto live = [&] { return device.stub->remoteAllocs - device.stub->remoteFrees; };
  for (size_t launch = 1; launch <= 8; ++launch) {
    Capture capture{};
    REQUIRE(polydco_dispatch(0, 9, 1, PlatformKind::Managed, &bundle, 0, nullptr, reinterpret_cast<char *>(&capture)));
    CHECK(device.stub->remoteAllocs >= launch);
    CHECK(live() == 0);
  }
}

#endif

TEST_CASE("skip probe: device lacks required features", "[.probe]") {
  WithStubDevice device(false);
  WithStubManagedPlatform platform;
  device.stub->advertised = std::vector<std::string>{"x87"};
  const unsigned char image = 0;
  std::array<const char *, 3> features{"int64", "fp64", "fp16"};
  const polyregion::runtime::KernelObject object{PlatformKind::Managed,
                                                 ModuleFormat::Object,
                                                 features.size(),
                                                 features.data(),
                                                 sizeof(image),
                                                 &image,
                                                 polyregion::compiletime::Target::Object_LLVM_HOST,
                                                 "",
                                                 "",
                                                 polyregion::compiletime::OptLevel::O3,
                                                 0,
                                                 nullptr};
  REQUIRE_FALSE(polyregion::polyrt::loadKernelObject("lacking", object));
  REQUIRE_FALSE(polyregion::polyrt::loadKernelObject("lacking-again", object));
  polyregion::polyrt::noCompatibleKernelExit(__func__);
}

TEST_CASE("skip probe: device format differs from every image", "[.probe]") {
  WithStubDevice device(false, false, 511, 64, ModuleFormat::SPIRV_GLCompute);
  WithStubManagedPlatform platform;
  const unsigned char image = 0;
  const polyregion::runtime::KernelObject object{PlatformKind::Managed,
                                                 ModuleFormat::Object,
                                                 0,
                                                 nullptr,
                                                 sizeof(image),
                                                 &image,
                                                 polyregion::compiletime::Target::Object_LLVM_HOST,
                                                 "",
                                                 "",
                                                 polyregion::compiletime::OptLevel::O3,
                                                 0,
                                                 nullptr};
  REQUIRE_FALSE(polyregion::polyrt::loadKernelObject("mismatch", object));
  polyregion::polyrt::noCompatibleKernelExit(__func__);
}

TEST_CASE("skip probe: nothing was offered", "[.probe]") { polyregion::polyrt::noCompatibleKernelExit(__func__); }

TEST_CASE("skip probe: no device", "[.probe]") {
  polyregion::polyrt::initialise();
  polyregion::polyrt::currentDevice.reset();
  const unsigned char image = 0;
  const polyregion::runtime::KernelObject object{PlatformKind::Managed,
                                                 ModuleFormat::Object,
                                                 0,
                                                 nullptr,
                                                 sizeof(image),
                                                 &image,
                                                 polyregion::compiletime::Target::Object_LLVM_HOST,
                                                 "",
                                                 "",
                                                 polyregion::compiletime::OptLevel::O3,
                                                 0,
                                                 nullptr};
  REQUIRE_FALSE(polyregion::polyrt::loadKernelObject("deviceless", object));
  polyregion::polyrt::noCompatibleKernelExit(__func__);
}
