#include "polystl/polystl.h"

#include "polyregion/conventions.h"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"
#include "polyrt/sma_runtime.hpp"

#include "reflect-rt/rt_reflect.hpp"

// XXX polystl-static bundles polyinvoke + LLVM Support; consumers link via polycpp at runtime
// without CMake propagation, so pull in the Windows system imports from this object.
#if defined(_MSC_VER)
  #pragma comment(lib, "Version.lib")
  #pragma comment(lib, "psapi.lib")
  #pragma comment(lib, "ntdll.lib")
  #pragma comment(lib, "ws2_32.lib")
  #pragma comment(lib, "ole32.lib")
  #pragma comment(lib, "shell32.lib")
  #pragma comment(lib, "advapi32.lib")
  #pragma comment(lib, "uuid.lib")
  #pragma comment(lib, "delayimp.lib")
#endif

using namespace polyregion;
using polyrt::DebugLevel;

POLYREGION_RT_PROTECT static polyrt::SynchronisedMemAllocation allocations(
    [](const void *ptr) -> polyrt::PtrQuery {
      if (const auto meta = rt_reflect::_rt_reflect_p(ptr); meta.type != rt_reflect::Type::Unknown) {
        return polyrt::PtrQuery{
            .sizeInBytes = meta.size, .offsetInBytes = meta.offset, .hostReadOnly = meta.type == rt_reflect::Type::StaticRodata};
      }
      polyrt::ensureRoSegmentsRecorded();
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

POLYREGION_EXPORT extern "C" void polyrt_map_read(void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  polyrt::sma::mapRead(allocations, origin, sizeInBytes, unitInBytes);
}

POLYREGION_EXPORT extern "C" void polyrt_map_write(void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  polyrt::sma::mapWrite(allocations, origin, sizeInBytes, unitInBytes);
}

POLYREGION_EXPORT extern "C" void polyrt_map_readwrite(void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  polyrt::sma::mapReadwrite(allocations, origin, sizeInBytes, unitInBytes);
}

POLYREGION_DEFINE_SMA_MIRROR_ABI(allocations);

POLYREGION_EXPORT void polystl::details::dispatchHostThreaded(const size_t global, void *functorData, const char *moduleId) {
  using namespace invoke;
  log(DebugLevel::Debug, "<%s:%s:%zu> Dispatch hostthread", __func__, moduleId, global);
  ArgBuffer buffer{{Type::IntS64, nullptr}, {Type::Ptr, &functorData}, {Type::Void, nullptr}};
  polyrt::currentQueue->enqueueInvokeAsync(moduleId, conventions::EntryName, buffer, Policy{Dim3{global, 1, 1}}, {});
  polyrt::currentQueue->enqueueWaitBlocking();
  log(DebugLevel::Debug, "<%s:%s:%zu> Done", __func__, moduleId, global);
}

POLYREGION_EXPORT void polystl::details::dispatchManaged(const size_t global, const size_t local, const size_t localMemBytes,
                                                         const runtime::TypeLayout *layout, void *functorData, const char *moduleId,
                                                         const runtime::PreludeFn prelude, const runtime::PostludeFn postlude) {
  using namespace invoke;

  log(DebugLevel::Debug, "<%s:%s:%zu> Dispatch managed, arg=%p bytes, prelude=%p", __func__, moduleId, global, functorData,
      reinterpret_cast<void *>(prelude));
  const auto launch = [&](const ArgBuffer &buffer) {
    polyrt::currentQueue->enqueueInvokeAsync(
        moduleId, conventions::EntryName, buffer, //
        Policy{Dim3{global, 1, 1}, local > 0 ? std::optional{std::pair{Dim3{local, 1, 1}, localMemBytes}} : std::nullopt}, {});
    log(DebugLevel::Debug, "<%s:%s:%zu> Submitted", __func__, moduleId, global);
    polyrt::currentQueue->enqueueWaitBlocking();
  };

  const auto mode = polyrt::sma::mirrorModeFor(polyrt::currentDevice->moduleFormat());
  if (mode == polyrt::sma::MirrorMode::Off) {
    if (polyrt::currentDevice->pagingMode() != PagingMode::System)
      polyrt::skipExit("POLYRT_MIRROR=off needs system paging (HMM / XNACK+ / system-USM)");
    launch(localMemBytes > 0 ? ArgBuffer{{Type::Scratch, {}}, {Type::Ptr, &functorData}, {Type::Void, {}}}
                             : ArgBuffer{{Type::Ptr, &functorData}, {Type::Void, {}}});
    log(DebugLevel::Debug, "<%s:%s:%zu> Done (usm)", __func__, moduleId, global);
    return;
  }

  if (mode == polyrt::sma::MirrorMode::Arena) {
    // lay the whole capture graph into one device arena (pointers -> arena offsets); the
    // arena-lowered kernel takes the arena base (capture root at offset 0) and derefs arena[off + i]
    allocations.genArenaReset();
    allocations.genArenaObjectSlack = invoke::overReadPadBytes(polyrt::currentDevice->features());
    allocations.genArenaMirror(static_cast<const char *>(functorData), 1, 1, *layout, layout->sizeInBytes);
    auto arenaBase = reinterpret_cast<void *>(allocations.genArenaFinish());
    // SPIR-V reads the arena through a fixed roster of typed views, so bind the one buffer to every view
    // slot; flat backends take it once
    const int arenaViews = polyrt::sma::arenaViewForm(polyrt::currentDevice->moduleFormat()) ? polyrt::sma::arenaViewCount : 1;
    ArgBuffer buffer;
    if (localMemBytes > 0) buffer.append(Type::Scratch, nullptr);
    for (int i = 0; i < arenaViews; ++i)
      buffer.append(Type::Ptr, &arenaBase);
    buffer.append(Type::Void, nullptr);
    launch(buffer);
    allocations.genArenaReadback();
    log(DebugLevel::Debug, "<%s:%s:%zu> Done (arena)", __func__, moduleId, global);
    return;
  }

  const bool useGenerated = mode == polyrt::sma::MirrorMode::Compiletime && prelude;
  const uintptr_t preludeResult = useGenerated ? prelude(functorData, layout->sizeInBytes) : 0;
  polyrt::sma::dumpAllocationTable(allocations);

  auto functorDevicePtr = useGenerated ? reinterpret_cast<void *>(preludeResult)
                                       : reinterpret_cast<void *>(allocations.syncLocalToRemote(functorData, *layout));
  const auto buffer = localMemBytes > 0 ? ArgBuffer{{Type::Scratch, {}}, {Type::Ptr, &functorDevicePtr}, {Type::Void, {}}}
                                        : ArgBuffer{{Type::Ptr, &functorDevicePtr}, {Type::Void, {}}};
  launch(buffer);
  // sync device-side writes to captured pointer-targets back before the device alloc is freed
  if (useGenerated && postlude) postlude(functorData, layout->sizeInBytes);
  else allocations.syncRemoteToLocal(functorData);
  allocations.disassociate(functorData);
  log(DebugLevel::Debug, "<%s:%s:%zu> Done", __func__, moduleId, global);
}
