#include "polydco.h"

#include "polyregion/concurrency_utils.hpp"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"

static std::unordered_map<uintptr_t, size_t> ptrRecords;
static std::mutex mutex;

using namespace polyregion;

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_record(void *ptr, size_t size) {
  std::unique_lock lock(mutex);
  fprintf(stderr, "polydco_record(%p, %ld)\n", ptr, size);
  ptrRecords[reinterpret_cast<uintptr_t>(ptr)] = size;
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_release(void *ptr) {
  std::unique_lock lock(mutex);
  fprintf(stderr, "polydco_release(%p)\n", ptr);
  ptrRecords.erase(reinterpret_cast<uintptr_t>(ptr));
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_debug_farraydesc(FArrayDesc *desc) {
  fprintf(stderr, "PolyDCOFArrayDesc{\n");
  fprintf(stderr, "  .addr = %p,\n", desc->addr);
  fprintf(stderr, "  .sizeInBytes = %luULL,\n", desc->sizeInBytes);
  fprintf(stderr, "  .ranks = %luULL,\n", desc->ranks);

  fprintf(stderr, "  .dims = new PolyDCOFDim[%lu]{\n", desc->ranks);
  for (uint64_t i = 0; i < desc->ranks; ++i) {
    fprintf(stderr, "      { .lowerBound = %luULL, .extent = %luULL, .stride = %luULL }", desc->dims[i].lowerBound, desc->dims[i].extent,
            desc->dims[i].stride);
    if (i + 1 < desc->ranks) fprintf(stderr, ",");
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "  }\n");
  fprintf(stderr, "};\n");
}

POLYREGION_EXPORT extern "C" [[maybe_unused]] void polydco_debug_typelayout(runtime::TypeLayout *layout) {
  layout->print(stderr);
  layout->visualise(stderr);
}

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

static void dispatchManaged(const int64_t lowerBoundInclusive, const int64_t upperBoundInclusive, const int64_t step, //
                            const size_t localBound, const size_t localMemBytes,                                      //
                            const runtime::TypeLayout *layout,                                                        //
                            char *captures, const char *moduleId) {
  struct Prelude {
    int64_t lowerBound, upperBound, step, tripCount;
  };
  static_assert(sizeof(Prelude) == (sizeof(int64_t) * 4));

  const auto upperBoundExclusive = upperBoundInclusive + 1;
  const int64_t tripCount = concurrency_utils::tripCountExclusive(lowerBoundInclusive, upperBoundExclusive, step);
  POLYDCO_LOG("<%s:%s:%zu> Dispatch managed, arg=%p bytes", __func__, moduleId, tripCount, captures);

  validatePrelude<Prelude>(layout, moduleId);

  const Prelude prelude{lowerBoundInclusive, upperBoundInclusive, step, tripCount};
  std::memcpy(captures, &prelude, sizeof(Prelude));

  const auto functorDevicePtr = allocations.syncLocalToRemote(captures, *layout);
  polyrt::dispatchManaged(tripCount, localBound, localMemBytes, reinterpret_cast<void *>(functorDevicePtr), moduleId);
  allocations.syncRemoteToLocal(captures);
  polyrt::currentQueue->enqueueWaitBlocking();
  POLYDCO_LOG("<%s:%s:%zu> Done", __func__, moduleId, tripCount);
}

static void dispatchHostThreaded(int64_t lowerBoundInclusive, int64_t upperBoundInclusive, int64_t step, //
                                 const runtime::TypeLayout *layout,                    //
                                 char *captures, const char *moduleId) {
  struct Prelude {
    int64_t lowerBound, upperBound, step, tripCount;
    int64_t *begins, *ends;
  };
  static_assert(sizeof(Prelude) == (sizeof(int64_t) * 4 + sizeof(void *) * 2));

  const auto upperBoundExclusive = upperBoundInclusive + 1;
  const int64_t tripCount = concurrency_utils::tripCountExclusive(lowerBoundInclusive, upperBoundExclusive, step);
  POLYDCO_LOG("<%s:%s:%zu> Dispatch host, arg=%p bytes", __func__, moduleId, tripCount, captures);

  validatePrelude<Prelude>(layout, moduleId);

  //  auto N = std::thread::hardware_concurrency();

  size_t N = 1;
  auto [begins, ends] = concurrency_utils::splitStaticExclusive<int64_t>(0, tripCount, N);
  const Prelude prelude{lowerBoundInclusive, upperBoundInclusive, step, tripCount, begins.data(), ends.data()};

  std::memcpy(captures, &prelude, sizeof(Prelude));

  polyrt::dispatchHostThreaded(N, captures, moduleId);
  polyrt::currentQueue->enqueueWaitBlocking();
  POLYDCO_LOG("<%s:%s:%zu> Done", __func__, moduleId, tripCount);
}

// bool fallback = false;
// if (currentPlatform == kind) {
//   // if(!dispatch(*layout, <create captures>))
//   //   fallback = true;
// } else { assert }
// if(fallback) do

bool polydco_is_platformkind(runtime::PlatformKind kind) {
  polyrt::initialise();
  return polyrt::currentPlatform->kind() == kind;
}

bool polydco_dispatch(const int64_t lowerBoundInclusive, const int64_t upperBoundInclusive, const int64_t step, //
                      runtime::PlatformKind kind,
                      const runtime::KernelBundle *bundle, //
                      char *captures) {
  polyrt::initialise();

  POLYDCO_LOG("<%s> Dispatch (%ld to %ld by %ld)", __func__, lowerBoundInclusive, upperBoundInclusive, step);

  if (!bundle || !captures) {
    POLYDCO_LOG("bundle=%p captures=%p, not dispatching", bundle, captures);
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

  size_t attempts = 0;
  switch (kind) {
    case runtime::PlatformKind::HostThreaded: {
      for (size_t i = 0; i < bundle->objectCount; ++i) {
        attempts++;
        if (!polyrt::loadKernelObject(bundle->moduleName, bundle->objects[i])) continue;
        dispatchHostThreaded(lowerBoundInclusive, upperBoundInclusive, step, &bundle->structs[bundle->interfaceLayoutIdx], captures,
                             bundle->moduleName);
        return true;
      }
      break;
    }
    case runtime::PlatformKind::Managed: {
      for (size_t i = 0; i < bundle->objectCount; ++i) {
        attempts++;
        if (!polyrt::loadKernelObject(bundle->moduleName, bundle->objects[i])) continue;
        dispatchManaged(lowerBoundInclusive, upperBoundInclusive, step, 0, 0, &bundle->structs[bundle->interfaceLayoutIdx], captures,
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
