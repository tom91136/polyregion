#include "polystl/polystl.h"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"
#include "reflect-rt//rt_reflect.hpp"

static polyregion::polyrt::SynchronisedMemAllocation allocations(
    [](const void *ptr) -> polyregion::polyrt::PtrQuery {
      if (const auto meta = polyregion::rt_reflect::_rt_reflect_p(ptr); meta.type != polyregion::rt_reflect::Type::Unknown) {
        return polyregion::polyrt::PtrQuery{.sizeInBytes = meta.size, .offsetInBytes = meta.offset};
      }
      POLYSTL_LOG("Local: Failed to query %p", ptr);
      return polyregion::polyrt::PtrQuery{0, 0};
    },
    /*remoteAlloc*/
    [](const size_t size) { //
      const auto p = polyregion::polyrt::currentDevice->mallocDevice(size, polyregion::invoke::Access::RW);
      POLYSTL_LOG("                               Remote 0x%jx = malloc(%ld)", p, size);
      return p;
    },
    /*remoteRead*/
    [](void *dst, const uintptr_t src, const size_t srcOffset, const size_t size) {
      POLYSTL_LOG("Local %p <|[%4ld]- Remote [%p + %4ld]", dst, size, reinterpret_cast<void *>(src), srcOffset);
      polyregion::polyrt::currentQueue->enqueueDeviceToHostAsync(src, srcOffset, dst, size, {});
    },
    /*remoteWrite*/
    [](const void *src, const uintptr_t dst, const size_t dstOffset, const size_t size) {
      POLYSTL_LOG("Local %p -[%4ld]|> Remote [%p + %4ld]", src, size, reinterpret_cast<void *>(dst), dstOffset);
      polyregion::polyrt::currentQueue->enqueueHostToDeviceAsync(src, dst, dstOffset, size, {});
    },
    /*remoteRelease*/
    [](const uintptr_t remotePtr) {
      POLYSTL_LOG("                               Remote free(0x%jx)", remotePtr);
      polyregion::polyrt::currentDevice->freeDevice(remotePtr);
    });

POLYREGION_EXPORT void polyregion::polystl::details::initialise() { polyrt::initialise(); }

POLYREGION_EXPORT void polyregion::polystl::details::dispatchManaged(const size_t global, const size_t local, const size_t localMemBytes,
                                                   const runtime::TypeLayout *layout, void *functorData, const char *moduleId) {
  POLYSTL_LOG("<%s:%s:%zu> Dispatch managed, arg=%p bytes", __func__, moduleId, global, functorData);
  const auto functorDevicePtr = allocations.syncLocalToRemote(functorData, *layout);
  polyrt::dispatchManaged(global, local, localMemBytes, reinterpret_cast<void *>(functorDevicePtr), moduleId);
  allocations.syncRemoteToLocal(functorData);
  polyrt::currentQueue->enqueueWaitBlocking();
  POLYSTL_LOG("<%s:%s:%zu> Done", __func__, moduleId, global);
}
