#include "polystl/polystl.h"

#include "polyregion/show.hpp"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"

#include "reflect-rt/rt_reflect.hpp"

#if !defined(_WIN32) && !defined(__APPLE__)
  #include <mutex>
  #include <vector>

  #include <link.h>
#endif

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

#if !defined(_WIN32) && !defined(__APPLE__)
// XXX register main exe's RO PT_LOADs with polyreflect lazily; captures into .rodata otherwise
// hit SMA's foreign-pointer path and mix USM with host VAs on hsa/cuda.
struct RoSegment {
  uintptr_t base;
  size_t size;
};
POLYREGION_RT_PROTECT static int collectRoSegment(struct dl_phdr_info *info, size_t, void *data) {
  if (info->dlpi_name && info->dlpi_name[0] != '\0') return 0;
  auto &out = *static_cast<std::vector<RoSegment> *>(data);
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const auto &ph = info->dlpi_phdr[i];
    if (ph.p_type != PT_LOAD) continue;
    if ((ph.p_flags & PF_W) != 0) continue;
    if ((ph.p_flags & PF_X) != 0) continue;
    if (ph.p_memsz == 0) continue;
    out.push_back({info->dlpi_addr + ph.p_vaddr, ph.p_memsz});
  }
  return 0;
}
POLYREGION_RT_PROTECT static std::vector<RoSegment> roSegments;
POLYREGION_RT_PROTECT static std::once_flag roSegmentsOnce;
POLYREGION_RT_PROTECT static void ensureRoSegmentsRecorded() {
  std::call_once(roSegmentsOnce, [] {
    dl_iterate_phdr(collectRoSegment, &roSegments);
    for (const auto &s : roSegments)
      rt_reflect::_rt_record(reinterpret_cast<void *>(s.base), s.size, rt_reflect::Type::StaticRodata);
  });
}
#endif

POLYREGION_RT_PROTECT static polyrt::SynchronisedMemAllocation allocations(
    [](const void *ptr) -> polyrt::PtrQuery {
      if (const auto meta = rt_reflect::_rt_reflect_p(ptr); meta.type != rt_reflect::Type::Unknown) {
        return polyrt::PtrQuery{
            .sizeInBytes = meta.size, .offsetInBytes = meta.offset, .hostReadOnly = meta.type == rt_reflect::Type::StaticRodata};
      }
#if !defined(_WIN32) && !defined(__APPLE__)
      ensureRoSegmentsRecorded();
      if (const auto meta = rt_reflect::_rt_reflect_p(ptr); meta.type != rt_reflect::Type::Unknown) {
        return polyrt::PtrQuery{
            .sizeInBytes = meta.size, .offsetInBytes = meta.offset, .hostReadOnly = meta.type == rt_reflect::Type::StaticRodata};
      }
#endif
      log(DebugLevel::Debug, "Local: Failed to query %p", ptr);
      return polyrt::PtrQuery{0, 0};
    },
    /*remoteAlloc*/
    [](const size_t size) { //
      const auto p = polyrt::currentDevice->mallocDevice(size, invoke::Access::RW);
      log(DebugLevel::Debug, "                               Remote 0x%jx = malloc(%ld)", p, size);
      return p;
    },

    /*remoteRead*/
    [](void *dst, const uintptr_t src, const size_t srcOffset, const size_t size) {
      log(DebugLevel::Debug, "Local %p <|[%4ld]- Remote [%p + %4ld]", dst, size, reinterpret_cast<void *>(src), srcOffset);
      polyrt::currentQueue->enqueueDeviceToHostAsync(src, srcOffset, dst, size, {});
      polyrt::currentQueue->enqueueWaitBlocking();
    },
    /*remoteWrite*/
    [](const void *src, const uintptr_t dst, const size_t dstOffset, const size_t size) {
      log(DebugLevel::Debug, "Local %p -[%4ld]|> Remote [%p + %4ld]", src, size, reinterpret_cast<void *>(dst), dstOffset);
      polyrt::currentQueue->enqueueHostToDeviceAsync(src, dst, dstOffset, size, {});
      polyrt::currentQueue->enqueueWaitBlocking();
    },
    /*remoteRelease*/
    [](const uintptr_t remotePtr) {
      log(DebugLevel::Debug, "                               Remote free(0x%jx)", remotePtr);
      polyrt::currentDevice->freeDevice(remotePtr);
    },
    polyrt::debugLevel() >= DebugLevel::Debug);

POLYREGION_RT_PROTECT static void dumpAllocationTable() {
  log(DebugLevel::Debug, "[Allocations (%lu)]", allocations.localToRemoteAlloc.size());
  for (auto [k, v] : allocations.localToRemoteAlloc) {
    const auto &l = allocations.remoteToLocalPtr[v.remote.ptr];
    log(DebugLevel::Debug, "\t[Local(0x%jx, %4ld) -> Remote(0x%jx, %4ld) %10s]", //
        k, l.sizeInBytes, v.remote.ptr, v.remote.sizeInBytes, v.layout ? v.layout->name : "???");
  }
}

POLYREGION_RT_PROTECT static void dumpMemoryWithLayout(const runtime::TypeLayout *layout, const char *data) {
  layout->visualise(stderr, [&](const size_t offset, const runtime::AggregateMember &m) {
    const auto p = data + offset;
    std::fprintf(stderr, "  [%p]=", static_cast<const void *>(p));
    if (m.ptrIndirection != 0) compiletime::showPtr(stderr, sizeof(void *), p);
    else switch (m.type->name[0]) {
        case 'I': compiletime::showInt(stderr, false, m.type->sizeInBytes, p); break;
        case 'U': compiletime::showInt(stderr, true, m.type->sizeInBytes, p); break;
        case 'F': compiletime::showFloat(stderr, m.type->sizeInBytes, p); break;
        default: compiletime::showHex(stderr, m.type->sizeInBytes, p); break;
      }
  });
}

POLYREGION_RT_PROTECT static void dumpAllocations() {
  for (auto [k, v] : allocations.localToRemoteAlloc) {
    if (!v.layout || v.layout->memberCount == 0) continue;
    dumpMemoryWithLayout(v.layout, reinterpret_cast<const char *>(k));
  }
}

POLYREGION_EXPORT extern "C" void polyrt_map_read(void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  log(DebugLevel::Debug, "polyrt_map_read(%p, %ld, %ld)", origin, sizeInBytes, unitInBytes);
  if (sizeInBytes == 0) return;
  allocations.syncRemoteToLocal(origin, sizeInBytes);
  dumpAllocations();
}

POLYREGION_EXPORT extern "C" void polyrt_map_write(void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  log(DebugLevel::Debug, "polyrt_map_write(%p, %ld, %ld)", origin, sizeInBytes, unitInBytes);
  if (sizeInBytes == 0) return;
  // allocations.syncRemoteToLocal(origin);
  allocations.invalidateLocal(origin);
}

POLYREGION_EXPORT extern "C" void polyrt_map_readwrite(void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  log(DebugLevel::Debug, "polyrt_map_readwrite(%p, %ld, %ld)", origin, sizeInBytes, unitInBytes);
  if (sizeInBytes == 0) return;
  allocations.syncRemoteToLocal(origin);
  allocations.invalidateLocal(origin);
}

POLYREGION_EXPORT void polystl::details::dispatchHostThreaded(const size_t global, void *functorData, const char *moduleId) {
  using namespace invoke;
  log(DebugLevel::Debug, "<%s:%s:%zu> Dispatch hostthread", __func__, moduleId, global);
  ArgBuffer buffer{{Type::IntS64, nullptr}, {Type::Ptr, &functorData}, {Type::Void, nullptr}};
  polyrt::currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer, Policy{Dim3{global, 1, 1}}, {});
  polyrt::currentQueue->enqueueWaitBlocking();
  log(DebugLevel::Debug, "<%s:%s:%zu> Done", __func__, moduleId, global);
}

POLYREGION_EXPORT void polystl::details::dispatchManaged(const size_t global, const size_t local, const size_t localMemBytes,
                                                         const runtime::TypeLayout *layout, void *functorData, const char *moduleId) {
  using namespace invoke;

  log(DebugLevel::Debug, "<%s:%s:%zu> Dispatch managed, arg=%p bytes", __func__, moduleId, global, functorData);
  auto functorDevicePtr = reinterpret_cast<void *>(allocations.syncLocalToRemote(functorData, *layout));
  // auto functorDevicePtr = reinterpret_cast<void *>(functorData);

  dumpAllocationTable();
  const auto buffer = localMemBytes > 0 ? ArgBuffer{{Type::Scratch, {}}, {Type::Ptr, &functorDevicePtr}, {Type::Void, {}}}
                                        : ArgBuffer{{Type::Ptr, &functorDevicePtr}, {Type::Void, {}}};
  polyrt::currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer, //
                                           Policy{                    //
                                                  Dim3{global, 1, 1}, //
                                                  local > 0 ? std::optional{std::pair{Dim3{local, 1, 1}, localMemBytes}} : std::nullopt},
                                           {});
  log(DebugLevel::Debug, "<%s:%s:%zu> Submitted", __func__, moduleId, global);
  polyrt::currentQueue->enqueueWaitBlocking();
  // Sync device-side writes to captured pointer-targets back before the device alloc is freed.
  allocations.syncRemoteToLocal(functorData);
  allocations.disassociate(functorData);
  log(DebugLevel::Debug, "<%s:%s:%zu> Done", __func__, moduleId, global);
}
