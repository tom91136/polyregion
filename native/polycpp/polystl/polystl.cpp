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

POLYREGION_EXPORT void polystl::details::dispatchHostThreaded(const size_t global, void *functorData, const char *moduleId) {
  using namespace invoke;
  log(DebugLevel::Debug, "<%s:%s:%zu> Dispatch hostthread", __func__, moduleId, global);
  ArgBuffer buffer{{Type::IntS64, nullptr}, {Type::Ptr, &functorData}, {Type::Void, nullptr}};
  polyrt::currentQueue->enqueueInvokeAsync(moduleId, conventions::EntryName, buffer, Policy{Dim3{global, 1, 1}}, {});
  polyrt::currentQueue->enqueueWaitBlocking();
  log(DebugLevel::Debug, "<%s:%s:%zu> Done", __func__, moduleId, global);
}

POLYREGION_EXPORT void polystl::details::dispatchManaged(const size_t global, const size_t local, const size_t localMemBytes,
                                                         const runtime::TypeLayout *layout, void *functorData, const char *moduleId) {
  using namespace invoke;

  log(DebugLevel::Debug, "<%s:%s:%zu> Dispatch managed, arg=%p bytes", __func__, moduleId, global, functorData);
  auto functorDevicePtr = reinterpret_cast<void *>(allocations.syncLocalToRemote(functorData, *layout));
  // auto functorDevicePtr = reinterpret_cast<void *>(functorData);

  polyrt::sma::dumpAllocationTable(allocations);
  const auto buffer = localMemBytes > 0 ? ArgBuffer{{Type::Scratch, {}}, {Type::Ptr, &functorDevicePtr}, {Type::Void, {}}}
                                        : ArgBuffer{{Type::Ptr, &functorDevicePtr}, {Type::Void, {}}};
  polyrt::currentQueue->enqueueInvokeAsync(moduleId, conventions::EntryName, buffer, //
                                           Policy{                                   //
                                                  Dim3{global, 1, 1},                //
                                                  local > 0 ? std::optional{std::pair{Dim3{local, 1, 1}, localMemBytes}} : std::nullopt},
                                           {});
  log(DebugLevel::Debug, "<%s:%s:%zu> Submitted", __func__, moduleId, global);
  polyrt::currentQueue->enqueueWaitBlocking();
  // Sync device-side writes to captured pointer-targets back before the device alloc is freed.
  allocations.syncRemoteToLocal(functorData);
  allocations.disassociate(functorData);
  log(DebugLevel::Debug, "<%s:%s:%zu> Done", __func__, moduleId, global);
}
