#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "polyregion/show.hpp"
#include "polyregion/types.h"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"

namespace polyregion::polyrt::sma {

POLYREGION_RT_PROTECT inline auto stdRemoteAlloc() {
  return [](const size_t size) -> uintptr_t {
    const auto p = currentDevice->mallocDevice(size, invoke::Access::RW);
    log(DebugLevel::Debug, "                               Remote 0x%jx = malloc(%ld)", p, size);
    return p;
  };
}

POLYREGION_RT_PROTECT inline auto stdRemoteRead() {
  return [](void *dst, const uintptr_t src, const size_t srcOffset, const size_t size) {
    log(DebugLevel::Debug, "Local %p <|[%4ld]- Remote [%p + %4ld]", dst, size, reinterpret_cast<void *>(src), srcOffset);
    currentQueue->enqueueDeviceToHostAsync(src, srcOffset, dst, size, {});
    currentQueue->enqueueWaitBlocking();
  };
}

POLYREGION_RT_PROTECT inline auto stdRemoteWrite() {
  return [](const void *src, const uintptr_t dst, const size_t dstOffset, const size_t size) {
    log(DebugLevel::Debug, "Local %p -[%4ld]|> Remote [%p + %4ld]", src, size, reinterpret_cast<void *>(dst), dstOffset);
    currentQueue->enqueueHostToDeviceAsync(src, dst, dstOffset, size, {});
    currentQueue->enqueueWaitBlocking();
  };
}

POLYREGION_RT_PROTECT inline auto stdRemoteRelease() {
  return [](const uintptr_t remotePtr) {
    log(DebugLevel::Debug, "                               Remote free(0x%jx)", remotePtr);
    currentDevice->freeDevice(remotePtr);
  };
}

POLYREGION_RT_PROTECT inline void dumpMemoryWithLayout(const runtime::TypeLayout *layout, const char *data) {
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

template <typename SMA> POLYREGION_RT_PROTECT inline void dumpAllocationTable(SMA &sma) {
  log(DebugLevel::Debug, "[Allocations (%lu)]", sma.localToRemoteAlloc.size());
  for (auto [k, v] : sma.localToRemoteAlloc) {
    const auto &l = sma.remoteToLocalPtr[v.remote.ptr];
    log(DebugLevel::Debug, "\t[Local(0x%jx, %4ld) -> Remote(0x%jx, %4ld) %10s]", //
        k, l.sizeInBytes, v.remote.ptr, v.remote.sizeInBytes, v.layout ? v.layout->name : "???");
  }
}

template <typename SMA> POLYREGION_RT_PROTECT inline void dumpAllocations(SMA &sma) {
  for (auto [k, v] : sma.localToRemoteAlloc) {
    if (!v.layout || v.layout->memberCount == 0) continue;
    dumpMemoryWithLayout(v.layout, reinterpret_cast<const char *>(k));
  }
}

template <typename SMA>
POLYREGION_RT_PROTECT inline void mapRead(SMA &sma, void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  log(DebugLevel::Debug, "polyrt_map_read(%p, %ld, %ld)", origin, sizeInBytes, unitInBytes);
  if (sizeInBytes == 0) return;
  sma.syncRemoteToLocal(origin, sizeInBytes);
  dumpAllocations(sma);
}

template <typename SMA>
POLYREGION_RT_PROTECT inline void mapWrite(SMA &sma, void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  log(DebugLevel::Debug, "polyrt_map_write(%p, %ld, %ld)", origin, sizeInBytes, unitInBytes);
  if (sizeInBytes == 0) return;
  sma.invalidateLocal(origin);
}

template <typename SMA>
POLYREGION_RT_PROTECT inline void mapReadwrite(SMA &sma, void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  log(DebugLevel::Debug, "polyrt_map_readwrite(%p, %ld, %ld)", origin, sizeInBytes, unitInBytes);
  if (sizeInBytes == 0) return;
  sma.syncRemoteToLocal(origin);
  sma.invalidateLocal(origin);
}

} // namespace polyregion::polyrt::sma
