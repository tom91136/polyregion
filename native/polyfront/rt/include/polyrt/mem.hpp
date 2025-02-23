#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <unordered_map>

#include "polyregion/types.h"

#include <cmath>

namespace polyregion::polyrt {

struct PtrQuery {
  size_t sizeInBytes;
  size_t offsetInBytes;
};

template <typename LocalReflect,  //
          typename RemoteAlloc,   //
          typename RemoteRead,    //
          typename RemoteWrite,   //
          typename RemoteRelease> //
class SynchronisedMemAllocation {

  LocalReflect localReflect;
  RemoteAlloc remoteAlloc;
  RemoteRead remoteRead;
  RemoteWrite remoteWrite;
  RemoteRelease remoteRelease;
  bool debug;

  static char *readPtrValue(const char *base) {
    char *ptr{};
    std::memcpy(&ptr, base, sizeof(uintptr_t));
    return ptr;
  }

  template <typename M, typename F>
  static std::optional<std::pair<typename M::iterator, size_t>> offsetQueryIt(M &map, const void *p, F selectSize) {
    const auto localPtr = reinterpret_cast<uintptr_t>(p);
    if (const auto it = map.find(localPtr); it != map.end()) return std::pair{it, 0};
    for (auto it = map.begin(); it != map.end(); ++it) {
      const uintptr_t base = it->first;
      if (const size_t size = selectSize(it->second); localPtr >= base && localPtr < base + size) {
        return std::pair{it, localPtr - base};
      }
    }
    return {};
  }

  template <typename V, typename F>
  static std::optional<std::pair<V *, size_t>> offsetQuery(std::unordered_map<uintptr_t, V> &map, const void *p, F selectSize) {
    if (const auto query = offsetQueryIt(map, p, selectSize)) {
      auto [it, offset] = *query;
      return std::pair{&it->second, offset};
    }
    return {};
  }

  uintptr_t createDeviceAllocation(const char *local, size_t sizeInBytes, const runtime::TypeLayout *s) {
    const uintptr_t remotePtr = remoteAlloc(sizeInBytes);
    localToRemoteAlloc.emplace(reinterpret_cast<uintptr_t>(local),
                               RemoteAllocation{.layout = s, //
                                                .localModified = false,
                                                .remote = RangedAllocation{.ptr = remotePtr, .sizeInBytes = sizeInBytes}});
    remoteToLocalPtr.emplace(remotePtr, RangedAllocation{.ptr = reinterpret_cast<uintptr_t>(local), .sizeInBytes = sizeInBytes});
    return remotePtr;
  }

  void writeSubObject(const char *hostData, const size_t memberOffsetInBytes, const size_t indirection, const uintptr_t devicePtr,
                      const runtime::TypeLayout &tl, const runtime::AggregateMember::ResolvePtrSize resolvePtrSizeInBytes,
                      const auto effectiveSizeToCount) {
    if (debug)
      std::fprintf(stderr,
                   "[SMA] writeSubObject(hostData=%p, memberOffsetInBytes=%zu, indirection=%zu, devicePtr=%jx, t=@%s (sizeInBytes=%zd, "
                   "members=%zd))\n", //
                   static_cast<const void *>(hostData), memberOffsetInBytes, indirection, devicePtr, tl.name, tl.sizeInBytes,
                   tl.memberCount);
    if (char *memberLocalPtr = readPtrValue(hostData + memberOffsetInBytes); !memberLocalPtr) {
      constexpr uintptr_t nullValue = 0;
      remoteWrite(&nullValue, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else if (resolvePtrSizeInBytes) {
      const size_t sizeInBytes = resolvePtrSizeInBytes(hostData + memberOffsetInBytes);
      if (debug)
        std::fprintf(stderr, "[SMA]   localResolve<%p>(%p) = {size=%ld}\n", //
                     (void *)resolvePtrSizeInBytes, static_cast<void *>(memberLocalPtr), sizeInBytes);
      const uintptr_t memberRemotePtr = mirrorToRemote(memberLocalPtr, indirection, effectiveSizeToCount(sizeInBytes), tl);
      remoteWrite(&memberRemotePtr, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else if (const PtrQuery meta = localReflect(memberLocalPtr); meta.sizeInBytes != 0) {
      if (debug)
        std::fprintf(stderr, "[SMA]   localReflect(%p) = {size=%ld, offset=%ld}\n", static_cast<void *>(memberLocalPtr), meta.sizeInBytes,
                     meta.offsetInBytes); //
      const size_t effectiveSize = meta.sizeInBytes - meta.offsetInBytes;
      const uintptr_t memberRemotePtr = mirrorToRemote(memberLocalPtr, indirection, effectiveSizeToCount(effectiveSize), tl);
      remoteWrite(&memberRemotePtr, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else {
      std::fprintf(stderr, "[SMA] Warning: encountered foreign pointer %p\n", static_cast<void *>(memberLocalPtr));
    }
  }

  uintptr_t writeIndirect(const char *hostData, const uintptr_t devicePtr, const size_t ptrIndirections, const size_t count,
                          const runtime::TypeLayout &tl) {
    if (debug)
      std::fprintf(stderr, "[SMA] writeIndirect(hostData=%p, count=%ld,devicePtr=%jx, t=@ %s, ptrIndirections=%ld)\n", //
                   static_cast<const void *>(hostData), count, devicePtr, tl.name, ptrIndirections);
    for (size_t idx = 0; idx < count; ++idx) {
      const size_t componentSize = ptrIndirections - 1 == 1 ? tl.sizeInBytes : sizeof(uintptr_t);
      writeSubObject(hostData, idx * sizeof(uintptr_t), ptrIndirections - 1, devicePtr, tl, nullptr,
                     [&](const size_t effectiveSize) { return effectiveSize / componentSize; });
    }
    return devicePtr;
  }

  uintptr_t writeStruct(const char *hostData, const size_t offsetInBytes, const uintptr_t devicePtr, const size_t count,
                        const runtime::TypeLayout &tl, const bool writeBody) {
    if (debug)
      std::fprintf(stderr,
                   "[SMA] writeStruct(hostData=%p, offsetInBytes=%zu, count=%ld,devicePtr=%jx, t=@%s (sizeInBytes=%zd, members=%zd), "
                   "writeBody=%d)\n", //
                   static_cast<const void *>(hostData), offsetInBytes, count, devicePtr, tl.name, tl.sizeInBytes, tl.memberCount,
                   writeBody);
    if (writeBody) remoteWrite(hostData + offsetInBytes, devicePtr, offsetInBytes, tl.sizeInBytes * count);
    for (size_t idx = 0; idx < count; ++idx) {
      for (size_t m = 0; m < tl.memberCount; ++m) {
        const auto member = tl.members[m];

        const auto memberOffsetInBytes = offsetInBytes + (idx * tl.sizeInBytes + member.offsetInBytes);
        if (member.sizeInBytes == 0) continue;
        if (member.ptrIndirection > 0) {
          const size_t componentSize = member.ptrIndirection == 1 ? member.componentSize : member.sizeInBytes;
          writeSubObject(hostData, memberOffsetInBytes, member.ptrIndirection, devicePtr, *member.type, member.resolvePtrSizeInBytes,
                         [&](const size_t effectiveSize) { return effectiveSize / componentSize; });
        } else if (member.type->memberCount > 0) {
          writeStruct(hostData, memberOffsetInBytes, devicePtr, 1, *member.type, false);
        }
      }
    }
    return devicePtr;
  }

  uintptr_t mirrorToRemote(const char *hostData, const size_t ptrIndirections, const size_t count, const runtime::TypeLayout &l) {
    if (debug)
      std::fprintf(stderr, "[SMA] mirrorToRemote(hostData=%p, ptrIndirections=%zu, count=%zu, t=@%s)\n", //
                   static_cast<const void *>(hostData), ptrIndirections, count, l.name);
    if (const auto query = offsetQuery(localToRemoteAlloc, hostData, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      if (!alloc->localModified) {
        if (debug) std::fprintf(stderr, "[SMA]   hit (%p = %4ld + %4ld)\n", alloc->remote.ptr, alloc->remote.sizeInBytes, offsetInBytes);
        return alloc->remote.ptr + offsetInBytes;
      } else
        return ptrIndirections > 1 ? writeIndirect(hostData, alloc->remote.ptr, ptrIndirections, count, *alloc->layout)
                                   : writeStruct(hostData, 0, alloc->remote.ptr, count, *alloc->layout, true);
    } else
      return ptrIndirections > 1
                 ? writeIndirect(hostData, createDeviceAllocation(hostData, sizeof(uintptr_t) * count, nullptr), ptrIndirections, count, l)
                 : writeStruct(hostData, 0, createDeviceAllocation(hostData, l.sizeInBytes * count, &l), count, l, true);
  }

  void readSubObject(char *p, const size_t memberOffsetInBytes) {
    if (debug)
      std::fprintf(stderr, "[SMA] readSubObject(p=%p, memberOffsetInBytes=%zu)\n", static_cast<const void *>(p), memberOffsetInBytes);
    if (char *memberRemotePtr = readPtrValue(p + memberOffsetInBytes); !memberRemotePtr) {
      std::memset(p + memberOffsetInBytes, 0, sizeof(uintptr_t));
    } else if (const auto query = offsetQuery(remoteToLocalPtr, memberRemotePtr, [](auto &x) { return x.sizeInBytes; })) {
      auto [alloc, _] = *query;
      syncRemoteToLocal(reinterpret_cast<void *>(alloc->ptr));
      std::memcpy(p + memberOffsetInBytes, &alloc->ptr, sizeof(uintptr_t));
    } else {
      std::fprintf(stderr, "[SMA] Warning: remote introduced foreign member pointer %p\n", static_cast<void *>(memberRemotePtr));
    }
  }

  void readStruct(char *p, const size_t offsetInBytes, const runtime::TypeLayout *tl) {
    if (debug)
      std::fprintf(stderr, "[SMA] readStruct(p=%p, offsetInBytes=%zu, t=@%s)\n", static_cast<const void *>(p), offsetInBytes, tl->name);
    for (size_t m = 0; m < tl->memberCount; ++m) {
      const auto member = tl->members[m];
      if (member.sizeInBytes == 0) continue;
      if (member.ptrIndirection > 0) readSubObject(p, offsetInBytes + member.offsetInBytes);
      else if (member.type->memberCount > 0) readStruct(p, offsetInBytes + member.offsetInBytes, member.type);
    }
  }

  template <typename N> static constexpr std::enable_if_t<std::is_integral_v<N>, N> integralCeil(N x, N y) { return (x + y - 1) / y; }
  template <typename N> static constexpr std::enable_if_t<std::is_integral_v<N>, N> integralFloor(N x, N y) { return x / y; }

public:
  struct RangedAllocation {
    uintptr_t ptr;
    size_t sizeInBytes;
  };

  struct RemoteAllocation {
    const runtime::TypeLayout *layout;
    bool localModified;
    RangedAllocation remote;
  };

  std::unordered_map<uintptr_t, RemoteAllocation> localToRemoteAlloc{};
  std::unordered_map<uintptr_t, RangedAllocation> remoteToLocalPtr{};

  SynchronisedMemAllocation(LocalReflect localReflect, //
                            RemoteAlloc remoteAlloc,   //
                            RemoteRead remoteRead,     //
                            RemoteWrite remoteWrite,   //
                            RemoteRelease remoteRelease, const bool debug = false)
      : localReflect(localReflect),   //
        remoteAlloc(remoteAlloc),     //
        remoteRead(remoteRead),       //
        remoteWrite(remoteWrite),     //
        remoteRelease(remoteRelease), //
        debug(debug) {}

  [[nodiscard]] uintptr_t syncLocalToRemote(const void *p, const runtime::TypeLayout &s) {
    return mirrorToRemote(static_cast<const char *>(p), 1, 1, s);
  }

  std::optional<uintptr_t> syncLocalToRemote(const void *p) {
    if (debug) std::fprintf(stderr, "[SMA] syncLocalToRemote(p=%p)\n", p);
    if (const auto query = offsetQuery(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      if (!alloc->localModified) return alloc->remote.ptr + offsetInBytes;
      else
        return writeStruct(static_cast<const char *>(p), 0, alloc->remote.ptr, alloc->remote.sizeInBytes / alloc->layout->sizeInBytes,
                           *alloc->layout, true);
    }
    return {};
  }

  std::optional<uintptr_t> invalidateLocal(const void *p) { // any valid pointer, not just base
    if (debug) std::fprintf(stderr, "[SMA] invalidateLocal(p=%p)\n", p);
    if (const auto query = offsetQuery(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      alloc->localModified = true;
      return alloc->remote.ptr + offsetInBytes;
    }
    return {};
  }

  std::optional<uintptr_t> syncRemoteToLocal(void *p, const std::optional<size_t> &sizeInByte = {}) { // any valid pointer, not just base
    if (const auto query = offsetQuery(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      if (debug)
        std::fprintf(stderr, "[SMA] syncRemoteToLocal(p=%p, remote=%p, sizeInByte=%ld, offsetInBytes=%ld)\n", p, alloc->remote.ptr,
                     alloc->remote.sizeInBytes, offsetInBytes);
      // we want to perform an inclusive copy: an object is copied if its range intersects with the request range
      const runtime::TypeLayout *tl = alloc->layout;

      const size_t objSize = tl ? tl->sizeInBytes : sizeof(uintptr_t);
      const size_t objIdxOffset = offsetInBytes % objSize; // offset to return to base at p
      char *baseAtObjIdx = static_cast<char *>(p) - objIdxOffset;

      const size_t objIdxBegin = integralFloor(offsetInBytes, objSize);
      const size_t objIdxEnd = objIdxBegin + integralCeil(sizeInByte.value_or(objSize), objSize);
      const size_t totalObjBytes = (objIdxEnd - objIdxBegin) * objSize;
      remoteRead(baseAtObjIdx, alloc->remote.ptr, objIdxBegin * objSize, totalObjBytes);
      if (tl) {
        if (!isSet(tl->attrs, runtime::LayoutAttrs::Opaque)) { // short-circuit if the struct is opaque (no further pointers)
          for (size_t i = objIdxBegin; i < objIdxEnd; ++i)
            readStruct(baseAtObjIdx, i * objSize, tl);
        }
      } else { // array of pointers
        for (size_t i = objIdxBegin; i < objIdxEnd; ++i)
          readSubObject(baseAtObjIdx, i * objSize);
      }
      if (debug) std::fprintf(stderr, "[SMA] ok\n");
      return alloc->remote.ptr + offsetInBytes;
    }
    return {};
  }

  void disassociate(const void *p) { // any valid pointer, not just base
    if (const auto query = offsetQueryIt(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [it, offsetInBytes] = *query;
      if (debug)
        std::fprintf(stderr, "[SMA] disassociate(host=%jx, remote=%jx, size=%ld)\n", it->first, it->second.remote.ptr,
                     it->second.remote.sizeInBytes);
      remoteRelease(it->second.remote.ptr);
      remoteToLocalPtr.erase(it->second.remote.ptr);
      localToRemoteAlloc.erase(it);
    }
  }
};

} // namespace polyregion::polyrt