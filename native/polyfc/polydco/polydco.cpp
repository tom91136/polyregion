#include "polydco.h"

#include <cinttypes>
#include <span>
#include <thread>

#include "magic_enum/magic_enum.hpp"

#include "polyregion/concurrency_utils.hpp"
#include "polyregion/conventions.h"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"
#include "polyrt/sma_runtime.hpp"

#include "ftypes.h"
#include "reflect-rt/rt_reflect.hpp"

// XXX Intentionally leaked: the SMA destructor below runs free callbacks at shutdown that
// route through these maps; file-scope statics would already be gone by then.
static auto &ptrRecords = *new std::unordered_map<uintptr_t, size_t>();
static auto &recordsMutex = *new std::mutex();

using namespace polyregion;
using polyrt::DebugLevel;

static polyrt::SynchronisedMemAllocation allocations(
    [](const void *ptr) -> polyrt::PtrQuery {
      if (const auto it = ptrRecords.find(reinterpret_cast<uintptr_t>(ptr)); it != ptrRecords.end()) {
        return polyrt::PtrQuery{.sizeInBytes = it->second, .offsetInBytes = 0};
      }
      // XXX fall through to polyreflect-rt for captures we didn't record locally (libc calloc paths).
      if (const auto meta = rt_reflect::_rt_reflect_p(ptr); meta.type != rt_reflect::Type::Unknown) {
        return polyrt::PtrQuery{
            .sizeInBytes = meta.size, .offsetInBytes = meta.offset, .hostReadOnly = meta.type == rt_reflect::Type::StaticRodata};
      }
      log(DebugLevel::Debug, "Local: Failed to query %p", ptr);
      return polyrt::PtrQuery{0, 0};
    },
    polyrt::sma::stdRemoteAlloc(),   //
    polyrt::sma::stdRemoteRead(),    //
    polyrt::sma::stdRemoteWrite(),   //
    polyrt::sma::stdRemoteRelease(), //
    polyrt::debugLevel() >= DebugLevel::Debug);

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_record(void *ptr, const size_t size) {
  std::unique_lock lock(recordsMutex);
  log(DebugLevel::Debug, "polydco_record(%p, %ld)", ptr, size);
  ptrRecords[reinterpret_cast<uintptr_t>(ptr)] = size;
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_release(void *ptr) {
  std::unique_lock lock(recordsMutex);
  log(DebugLevel::Debug, "polydco_release(%p)", ptr);
  ptrRecords.erase(reinterpret_cast<uintptr_t>(ptr));
  // XXX Keep the SVM mirror; Intel coarse-grain clSVMAlloc reuses freed addresses and collides
  // with still-live allocs in the same dispatch chain.
  allocations.disassociate(ptr, /*releaseRemote*/ false);
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_debug_typelayout(const runtime::TypeLayout *layout) {
  layout->print(stderr);
  layout->visualise(stderr);
}

POLYREGION_EXPORT extern "C" void polyrt_map_read(void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  polyrt::sma::mapRead(allocations, origin, sizeInBytes, unitInBytes);
}

POLYREGION_EXPORT extern "C" void polyrt_map_write(void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  polyrt::sma::mapWrite(allocations, origin, sizeInBytes, unitInBytes);
}

POLYREGION_EXPORT extern "C" void polyrt_map_readwrite(void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  polyrt::sma::mapReadwrite(allocations, origin, sizeInBytes, unitInBytes);
}

template <typename T> static void validatePrelude(const runtime::TypeLayout *layout, const char *moduleId) {
  if (layout->memberCount < 1) {
    log(DebugLevel::Debug, "<%s> Struct layout has no members", moduleId);
    std::fflush(stderr);
    std::abort();
  }
  if (layout->members[0].offsetInBytes != 0 || layout->members[0].sizeInBytes != sizeof(T)) {
    log(DebugLevel::Debug, "<%s> Prelude layout mismatch on first member, layout is:", moduleId);
    layout->print(stderr);
    std::fflush(stderr);
    std::abort();
  }
}

struct ManagedPartialReduction {

  struct Allocation {
    uintptr_t ptr;
    size_t size;
  };

  const std::span<const polydco::FReduction> &reductions;
  int64_t range;
  uintptr_t devicePartials;
  std::vector<Allocation> allocations;

  ManagedPartialReduction(const std::span<const polydco::FReduction> &reductions, const int64_t range)
      : reductions(reductions), range(range), devicePartials(), allocations(reductions.size()) {}

  size_t allocatePartialsAsync() {
    if (reductions.empty()) return 0;
    devicePartials = polyrt::currentDevice->mallocDevice(sizeof(void *) * reductions.size(), polyrt::Access::RW);
    size_t localMemBytes = 0;
    for (size_t i = 0; i < reductions.size(); ++i) {
      const auto size = reductions[i].typeSizeInBytes() * range;
      const auto devicePtr = polyrt::currentDevice->mallocDevice(size, polyrt::Access::RW);
      allocations[i] = Allocation{.ptr = devicePtr, .size = size};
      polyrt::currentQueue->enqueueHostToDeviceAsync(&devicePtr, devicePartials, sizeof(void *) * i, sizeof(void *), {});
      polyrt::currentQueue->enqueueWaitBlocking();
      localMemBytes += size;
    }
    return localMemBytes;
  }

  void releaseAndReduce() const {
    if (reductions.empty()) return;
    polyrt::currentDevice->freeDevice(devicePartials);
    if (polyrt::currentDevice->sharedAddressSpace()) {
      for (size_t i = 0; i < allocations.size(); ++i)
        reductions[i].reduce(range, reinterpret_cast<const char *>(allocations[i].ptr));
    } else {
      std::vector<std::vector<char>> hostPartials(allocations.size());
      for (size_t i = 0; i < allocations.size(); ++i)
        hostPartials[i].resize(allocations[i].size);

      for (size_t i = 0; i < allocations.size(); ++i) {
        polyrt::currentQueue->enqueueDeviceToHostAsync(allocations[i].ptr, 0, hostPartials[i].data(), allocations[i].size, {});
      }
      polyrt::currentQueue->enqueueWaitBlocking();
      for (size_t i = 0; i < allocations.size(); ++i) {
        polyrt::currentDevice->freeDevice(allocations[i].ptr);
        reductions[i].reduce(range, hostPartials[i].data());
      }
    }
  }
};

static void dispatchManaged(const int64_t lowerBoundInclusive, const int64_t upperBoundInclusive, const int64_t step, //
                            const runtime::TypeLayout *layout,                                                        //
                            const std::span<const polydco::FReduction> &reductions,                                   //
                            char *captures, const char *moduleId) {

  const auto upperBoundExclusive = upperBoundInclusive + 1;
  const int64_t tripCount = concurrency_utils::tripCountExclusive(lowerBoundInclusive, upperBoundExclusive, step);
  log(DebugLevel::Debug, "<%s:%s:%" PRId64 "> Dispatch managed, arg=%p", __func__, moduleId, tripCount, static_cast<void *>(captures));

  validatePrelude<polydco::FManagedPrelude>(layout, moduleId);

  const polydco::FManagedPrelude prelude{lowerBoundInclusive, upperBoundInclusive, step, tripCount};
  std::memcpy(captures, &prelude, sizeof(polydco::FManagedPrelude));

  auto functorDevicePtr = allocations.syncLocalToRemote(captures, *layout);

  polyrt::sma::dumpAllocationTable(allocations);

  constexpr size_t blockSize = 256;
  const bool isReduction = !reductions.empty();
  const size_t threadsPerBlock = //
      isReduction ? blockSize    // use local==global for reduction
                  : (static_cast<size_t>(tripCount) < blockSize ? 1 : static_cast<size_t>(tripCount) / blockSize);
  const size_t blocks =       //
      isReduction ? blockSize // use local==global for reduction
                  : (static_cast<size_t>(tripCount) < blockSize ? static_cast<size_t>(tripCount) : blockSize);
  ManagedPartialReduction mpr(reductions, blockSize);
  const size_t localMemBytes = mpr.allocatePartialsAsync();
  log(DebugLevel::Debug, "<%s:%s:%zu> localMemBytes=%zu", __func__, moduleId, threadsPerBlock, localMemBytes);

  using namespace invoke;
  const auto buffer = isReduction ? ArgBuffer{{Type::Ptr, &functorDevicePtr},   //
                                              {Type::Ptr, &mpr.devicePartials}, //
                                              {Type::Scratch, {}},              //
                                              {Type::Void, {}}}                 //
                                  : ArgBuffer{                                  //
                                              {Type::Ptr, &functorDevicePtr},
                                              {Type::Ptr, &mpr.devicePartials},
                                              {Type::Void, {}}};

  log(DebugLevel::Debug, "<%s:%s:%zu> Dispatch managed, arg=%p managed=0x%jx", __func__, moduleId, threadsPerBlock,
      static_cast<void *>(captures), mpr.devicePartials);
  polyrt::currentQueue->enqueueInvokeAsync(moduleId, conventions::EntryName, buffer, //
                                           Policy{                                   //
                                                  Dim3{threadsPerBlock, 1, 1},       //
                                                  blocks > 0 ? std::optional{std::pair{Dim3{blocks, 1, 1}, localMemBytes}} : std::nullopt},
                                           {});

  log(DebugLevel::Debug, "<%s:%s:%zu> Submitted", __func__, moduleId, threadsPerBlock);
  polyrt::currentQueue->enqueueWaitBlocking();

  allocations.syncRemoteToLocal(captures);
  polyrt::currentQueue->enqueueWaitBlocking();

  // XXX See polydco_release: keep SVM, drop only host tracking.
  allocations.disassociate(captures, /*releaseRemote*/ false);

  mpr.releaseAndReduce();

  polyrt::currentQueue->enqueueWaitBlocking();
  polyrt::sma::dumpAllocations(allocations);
  log(DebugLevel::Debug, "<%s:%s:%" PRId64 "> Done", __func__, moduleId, tripCount);
}

static void dispatchHostThreaded(const int64_t lowerBoundInclusive, const int64_t upperBoundInclusive, const int64_t step, //
                                 const runtime::TypeLayout *layout,                                                        //
                                 const std::span<const polydco::FReduction> &reductions, char *captures, const char *moduleId) {

  const auto upperBoundExclusive = upperBoundInclusive + 1;
  const int64_t tripCount = concurrency_utils::tripCountExclusive(lowerBoundInclusive, upperBoundExclusive, step);
  log(DebugLevel::Debug, "<%s:%s:%" PRId64 "> Dispatch host, arg=%p", __func__, moduleId, tripCount, static_cast<void *>(captures));

  validatePrelude<polydco::FHostThreadedPrelude>(layout, moduleId);

  const size_t N = std::thread::hardware_concurrency();
  auto [begins, ends] = concurrency_utils::splitStaticExclusive<int64_t>(0, tripCount, N);
  const size_t groups = begins.size();

  const polydco::FHostThreadedPrelude prelude{lowerBoundInclusive, upperBoundInclusive, step, tripCount, begins.data(), ends.data()};
  std::memcpy(captures, &prelude, sizeof(polydco::FHostThreadedPrelude));

  if (polyrt::debugLevel() >= DebugLevel::Debug) {
    const auto preludeSize = sizeof(polydco::FHostThreadedPrelude);
    std::fprintf(stderr, "[HostTrace] captures=%p, layout->sizeInBytes=%zu, preludeSize=%zu\n", static_cast<void *>(captures),
                 layout->sizeInBytes, preludeSize);
    for (size_t i = 0; i < layout->memberCount; ++i) {
      std::fprintf(stderr, "[HostTrace] member[%zu] %s offset=%zu size=%zu", i, layout->members[i].name, layout->members[i].offsetInBytes,
                   layout->members[i].sizeInBytes);
      if (layout->members[i].sizeInBytes == sizeof(void *)) {
        void *ptr;
        std::memcpy(&ptr, captures + layout->members[i].offsetInBytes, sizeof(ptr));
        std::fprintf(stderr, " ptr=%p", ptr);
      }
      std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
  }

  ManagedPartialReduction mpr(reductions, groups);
  mpr.allocatePartialsAsync();
  using namespace invoke;
  const ArgBuffer buffer{{Type::IntS64, nullptr}, {Type::Ptr, &captures}, {Type::Ptr, &mpr.devicePartials}, {Type::Void, nullptr}};
  log(DebugLevel::Debug, "<%s:%s:%" PRId64 "> Dispatch hostthreaded", __func__, moduleId, tripCount);
  polyrt::currentQueue->enqueueInvokeAsync(moduleId, conventions::EntryName, buffer, Policy{Dim3{groups, 1, 1}},
                                           [&]() { mpr.releaseAndReduce(); });
  polyrt::currentQueue->enqueueWaitBlocking();
  if (polyrt::debugLevel() >= DebugLevel::Debug) {
    for (size_t i = 0; i < layout->memberCount; ++i) {
      if (layout->members[i].sizeInBytes == sizeof(void *)) {
        void *ptr;
        std::memcpy(&ptr, captures + layout->members[i].offsetInBytes, sizeof(ptr));
        if (ptr) {
          int32_t val32 = 0;
          std::memcpy(&val32, ptr, sizeof(val32));
          std::fprintf(stderr, "[HostTrace] post-kernel: %s ptr=%p *(i32)=%d\n", layout->members[i].name, ptr, val32);
        }
      }
    }
    std::fflush(stderr);
  }
  log(DebugLevel::Debug, "<%s:%s:%" PRId64 "> Done", __func__, moduleId, tripCount);
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] bool polydco_is_platformkind(const runtime::PlatformKind kind) {
  polyrt::initialise();
  // currentPlatform may be null when backend init fails (missing runtime library).
  return polyrt::currentPlatform && polyrt::currentPlatform->kind() == kind;
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] bool polydco_dispatch(const int64_t lowerBoundInclusive, const int64_t upperBoundInclusive,
                                                                    const int64_t step,                                                  //
                                                                    const runtime::PlatformKind kind,                                    //
                                                                    const runtime::KernelBundle *bundle,                                 //
                                                                    const size_t reductionsCount, const polydco::FReduction *reductions, //
                                                                    char *captures) {
  polyrt::initialise();

  log(DebugLevel::Debug, "<%s> Dispatch (%" PRId64 " to %" PRId64 " by %" PRId64 ")", __func__, lowerBoundInclusive, upperBoundInclusive,
      step);

  if (!bundle || !captures) {
    log(DebugLevel::Debug, "bundle=%p captures=%p, not dispatching", static_cast<const void *>(bundle), static_cast<void *>(captures));
    return false;
  }

  if (polyrt::debugLevel() >= DebugLevel::Debug) {

    for (size_t i = 0; i < bundle->structCount; ++i) {
      if (i == bundle->interfaceLayoutIdx) {
        log(DebugLevel::Debug, "Exported: %s", bundle->structs[i].name);
      }
      bundle->structs[i].visualise(stderr);
    }

    for (size_t i = 0; i < reductionsCount; ++i) {
      fprintf(stderr, "Reduction[%ld] = {%s, %s -> %p}\n", i, magic_enum::enum_name(reductions[i].type).data(),
              polydco::FReduction::to_string(reductions[i].kind).data(), static_cast<void *>(reductions[i].dest));
    }
  }

  size_t attempts = 0;
  switch (kind) {
    case runtime::PlatformKind::HostThreaded: {
      for (size_t i = 0; i < bundle->objectCount; ++i) {
        attempts++;
        if (!polyrt::loadKernelObject(bundle->moduleName, bundle->objects[i])) continue;
        dispatchHostThreaded(lowerBoundInclusive, upperBoundInclusive, step,
                             &bundle->structs[bundle->interfaceLayoutIdx],     //
                             std::span{reductions, reductionsCount}, captures, //
                             bundle->moduleName);
        return true;
      }
      break;
    }
    case runtime::PlatformKind::Managed: {
      for (size_t i = 0; i < bundle->objectCount; ++i) {
        attempts++;
        if (!polyrt::loadKernelObject(bundle->moduleName, bundle->objects[i])) continue;
        dispatchManaged(lowerBoundInclusive, upperBoundInclusive, step,
                        &bundle->structs[bundle->interfaceLayoutIdx],     //
                        std::span{reductions, reductionsCount}, captures, //
                        bundle->moduleName);
        return true;
      }
      break;
    }
  }

  if (!polyrt::hostFallback()) {
    // Exit 77 (autotools SKIP convention) so unsupported (backend, device, image) combos
    // are reported as skipped, not failed.
    log(DebugLevel::None,
        "Dispatch failed for %s: no compatible backend after trying %zu kernel objects, host fallback disabled - exiting 77 (skip)",
        bundle->moduleName, attempts);
    polyrt::noCompatibleKernelExit(polydco::abi::Dispatch);
  }

  return false;
}
