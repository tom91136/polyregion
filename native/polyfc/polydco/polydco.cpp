#include <span>

#include "ftypes.h"
#include "polydco.h"

#include "polyregion/concurrency_utils.hpp"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"

static std::unordered_map<uintptr_t, size_t> ptrRecords;
static std::mutex mutex;

using namespace polyregion;

static polyrt::SynchronisedAllocation allocations(
    [](const void *ptr) -> polyrt::PtrQuery {
      if (const auto it = ptrRecords.find(reinterpret_cast<uintptr_t>(ptr)); it != ptrRecords.end()) {
        return polyrt::PtrQuery{.sizeInBytes = it->second, .offsetInBytes = 0};
      }
      POLYDCO_LOG("Local: Failed to query %p", ptr);
      return polyrt::PtrQuery{0, 0};
    },
    /*remoteAlloc*/
    [](const size_t size) { //
      const auto p = polyrt::currentDevice->mallocDevice(size, invoke::Access::RW);
      POLYDCO_LOG("                               Remote 0x%jx = malloc(%ld)", p, size);
      return p;
    },
    /*remoteRead*/
    [](void *dst, const uintptr_t src, const size_t srcOffset, const size_t size) {
      POLYDCO_LOG("Local %p <|[%4ld]- Remote [%p + %4ld]", dst, size, reinterpret_cast<void *>(src), srcOffset);
      polyrt::currentQueue->enqueueDeviceToHostAsync(src, srcOffset, dst, size, {});
    },
    /*remoteWrite*/
    [](const void *src, const uintptr_t dst, const size_t dstOffset, const size_t size) {
      POLYDCO_LOG("Local %p -[%4ld]|> Remote [%p + %4ld]", src, size, reinterpret_cast<void *>(dst), dstOffset);
      polyrt::currentQueue->enqueueHostToDeviceAsync(src, dst, dstOffset, size, {});
    },
    /*remoteRelease*/
    [](const uintptr_t remotePtr) {
      POLYDCO_LOG("                               Remote free(0x%jx)", remotePtr);
      polyrt::currentDevice->freeDevice(remotePtr);
    });

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_record(void *ptr, const size_t size) {
  std::unique_lock lock(mutex);
  fprintf(stderr, "polydco_record(%p, %ld)\n", ptr, size);
  ptrRecords[reinterpret_cast<uintptr_t>(ptr)] = size;
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_release(void *ptr) {
  std::unique_lock lock(mutex);
  fprintf(stderr, "polydco_release(%p)\n", ptr);
  ptrRecords.erase(reinterpret_cast<uintptr_t>(ptr));
  allocations.disassociate(ptr);
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_debug_typelayout(const runtime::TypeLayout *layout) {
  layout->print(stderr);
  layout->visualise(stderr);
}

template <typename T> static void validatePrelude(const runtime::TypeLayout *layout, const char *moduleId) {
  if (layout->memberCount < 1) {
    POLYDCO_LOG("<%s> Struct layout has no members", moduleId);
    std::fflush(stderr);
    std::abort();
  }
  if (layout->members[0].offsetInBytes != 0 || layout->members[0].sizeInBytes != sizeof(T)) {
    POLYDCO_LOG("<%s> Prelude layout mismatch on first member, layout is:", moduleId);
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
      std::vector<void *> hostPartials(allocations.size());
      for (size_t i = 0; i < allocations.size(); ++i)
        hostPartials[i] = std::malloc(allocations[i].size);

      for (size_t i = 0; i < allocations.size(); ++i) {
        polyrt::currentQueue->enqueueDeviceToHostAsync(allocations[i].ptr, 0, hostPartials[i], allocations[i].size, {});
      }
      polyrt::currentQueue->enqueueWaitBlocking();
      for (size_t i = 0; i < allocations.size(); ++i) {
        polyrt::currentDevice->freeDevice(allocations[i].ptr);
        reductions[i].reduce(range, static_cast<const char *>(hostPartials[i]));
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
  POLYDCO_LOG("<%s:%s:%zu> Dispatch managed, arg=%p", __func__, moduleId, tripCount, static_cast<void *>(captures));

  validatePrelude<polydco::FManagedPrelude>(layout, moduleId);

  const polydco::FManagedPrelude prelude{lowerBoundInclusive, upperBoundInclusive, step, tripCount};
  std::memcpy(captures, &prelude, sizeof(polydco::FManagedPrelude));

  auto functorDevicePtr = allocations.syncLocalToRemote(captures, *layout);

  POLYDCO_LOG("[Allocations (%lu)]", allocations.localToRemoteAlloc.size());
  for (auto [k, v] : allocations.localToRemoteAlloc) {
    const auto &l = allocations.remoteToLocalPtr[v.remote.ptr];
    POLYDCO_LOG("\t[Local(%jx, %4ld) -> Remote(%jx, %4ld) %10s]", //
                k, l.sizeInBytes, v.remote.ptr, v.remote.sizeInBytes, v.layout->name);
  }

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
  POLYDCO_LOG("<%s:%s:%zu> localMemBytes=%ld", __func__, moduleId, threadsPerBlock, localMemBytes);

  using namespace invoke;
  const auto buffer = isReduction ? ArgBuffer{{Type::Ptr, &functorDevicePtr},   //
                                              {Type::Ptr, &mpr.devicePartials}, //
                                              {Type::Scratch, {}},              //
                                              {Type::Void, {}}}                 //
                                  : ArgBuffer{                                  //
                                              {Type::Ptr, &functorDevicePtr},
                                              {Type::Ptr, &mpr.devicePartials},
                                              {Type::Void, {}}};

  POLYDCO_LOG("<%s:%s:%zu> Dispatch managed, arg=%p managed=%jx", __func__, moduleId, threadsPerBlock, captures, mpr.devicePartials);
  polyrt::currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer,          //
                                           Policy{                             //
                                                  Dim3{threadsPerBlock, 1, 1}, //
                                                  blocks > 0 ? std::optional{std::pair{Dim3{blocks, 1, 1}, localMemBytes}} : std::nullopt},
                                           {});

  POLYDCO_LOG("<%s:%s:%zu> Submitted", __func__, moduleId, threadsPerBlock);
  polyrt::currentQueue->enqueueWaitBlocking();

  allocations.syncRemoteToLocal(captures);
  allocations.disassociate(captures);
  polyrt::currentQueue->enqueueWaitBlocking();

  mpr.releaseAndReduce();

  polyrt::currentQueue->enqueueWaitBlocking();
  POLYDCO_LOG("<%s:%s:%zu> Done", __func__, moduleId, tripCount);
}

static void dispatchHostThreaded(const int64_t lowerBoundInclusive, const int64_t upperBoundInclusive, const int64_t step, //
                                 const runtime::TypeLayout *layout,                                                        //
                                 const std::span<const polydco::FReduction> &reductions, char *captures, const char *moduleId) {

  const auto upperBoundExclusive = upperBoundInclusive + 1;
  const int64_t tripCount = concurrency_utils::tripCountExclusive(lowerBoundInclusive, upperBoundExclusive, step);
  POLYDCO_LOG("<%s:%s:%zu> Dispatch host, arg=%p", __func__, moduleId, tripCount, static_cast<void *>(captures));

  validatePrelude<polydco::FHostThreadedPrelude>(layout, moduleId);

  const size_t N = std::thread::hardware_concurrency();
  auto [begins, ends] = concurrency_utils::splitStaticExclusive<int64_t>(0, tripCount, N);
  const size_t groups = begins.size();

  const polydco::FHostThreadedPrelude prelude{lowerBoundInclusive, upperBoundInclusive, step, tripCount, begins.data(), ends.data()};
  std::memcpy(captures, &prelude, sizeof(polydco::FHostThreadedPrelude));

  ManagedPartialReduction mpr(reductions, groups);
  mpr.allocatePartialsAsync();
  using namespace invoke;
  const ArgBuffer buffer{{Type::IntS64, nullptr}, {Type::Ptr, &captures}, {Type::Ptr, &mpr.devicePartials}, {Type::Void, nullptr}};
  POLYDCO_LOG("<%s:%s:%zu> Dispatch hostthreaded", __func__, moduleId, tripCount);
  polyrt::currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer, Policy{Dim3{groups, 1, 1}}, [&]() { mpr.releaseAndReduce(); });
  polyrt::currentQueue->enqueueWaitBlocking();
  POLYDCO_LOG("<%s:%s:%zu> Done", __func__, moduleId, tripCount);
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] bool polydco_is_platformkind(const runtime::PlatformKind kind) {
  polyrt::initialise();
  return polyrt::currentPlatform->kind() == kind;
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] bool polydco_dispatch(const int64_t lowerBoundInclusive, const int64_t upperBoundInclusive,
                                                                    const int64_t step,                                                  //
                                                                    const runtime::PlatformKind kind,                                    //
                                                                    const runtime::KernelBundle *bundle,                                 //
                                                                    const size_t reductionsCount, const polydco::FReduction *reductions, //
                                                                    char *captures) {
  polyrt::initialise();

  POLYDCO_LOG("<%s> Dispatch (%ld to %ld by %ld)", __func__, lowerBoundInclusive, upperBoundInclusive, step);

  if (!bundle || !captures) {
    POLYDCO_LOG("bundle=%p captures=%p, not dispatching", static_cast<const void *>(bundle), static_cast<void *>(captures));
    return false;
  }

  for (size_t i = 0; i < bundle->structCount; i++) {
    const auto s = bundle->structs[i];
    s.print(stderr);
  }

  for (size_t i = 0; i < bundle->structCount; ++i) {
    if (i == bundle->interfaceLayoutIdx) fprintf(stderr, "**Exported**\n");
    bundle->structs[i].visualise(stderr);
  }

  for (size_t i = 0; i < reductionsCount; ++i) {
    fprintf(stderr, "Reduction[%ld] = {%s, %s -> %p}\n", i, to_string(reductions[i].type).data(),
            polydco::FReduction::to_string(reductions[i].kind).data(), reductions[i].dest);
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
    POLYDCO_LOG("Dispatch failed for %s: No compatible backend found after trying %zu different objects and host fallback is disabled, "
                "terminating program...",
                bundle->moduleName, attempts);
    std::fflush(stderr);
    std::abort();
  }

  return false;
}
