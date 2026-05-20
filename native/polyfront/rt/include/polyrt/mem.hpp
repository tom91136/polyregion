#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "polyregion/types.h"

namespace polyregion::polyrt {

struct PtrQuery {
  size_t sizeInBytes;
  size_t offsetInBytes;
  // XXX host page is mprotect'd RO (.rodata); SMA must skip device->host writeback or the copy faults.
  bool hostReadOnly = false;
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

  // XXX `allowPastEnd` enables matching `localPtr == base + size` (one-past-end). Off by default
  // because forward lookups in `mirrorToRemote` would otherwise attribute an unmapped allocation's
  // base to an adjacent mapped record's past-end.
  template <typename M, typename F>
  static std::optional<std::pair<typename M::iterator, size_t>> offsetQueryIt(M &map, const void *p, F selectSize,
                                                                              const bool allowPastEnd = false) {
    const auto localPtr = reinterpret_cast<uintptr_t>(p);
    if (const auto it = map.find(localPtr); it != map.end()) return std::pair{it, 0};
    typename M::iterator pastEnd = map.end();
    for (auto it = map.begin(); it != map.end(); ++it) {
      const uintptr_t base = it->first;
      const size_t size = selectSize(it->second);
      if (localPtr >= base && localPtr < base + size) return std::pair{it, localPtr - base};
      if (allowPastEnd && localPtr == base + size) pastEnd = it;
    }
    if (pastEnd != map.end()) return std::pair{pastEnd, localPtr - pastEnd->first};
    return {};
  }

  template <typename V, typename F>
  static std::optional<std::pair<V *, size_t>> offsetQuery(std::unordered_map<uintptr_t, V> &map, const void *p, F selectSize,
                                                           const bool allowPastEnd = false) {
    if (const auto query = offsetQueryIt(map, p, selectSize, allowPastEnd)) {
      auto [it, offset] = *query;
      return std::pair{&it->second, offset};
    }
    return {};
  }

  uintptr_t createDeviceAllocation(const char *local, size_t sizeInBytes, const runtime::TypeLayout *s, bool hostReadOnly = false) {
    // XXX +1 guard so adjacent USM allocs aren't bit-touching; otherwise A's past-end pointer aliases B's base in offsetQuery.
    const uintptr_t remotePtr = remoteAlloc(sizeInBytes + 1);
    localToRemoteAlloc.emplace(reinterpret_cast<uintptr_t>(local),
                               RemoteAllocation{.layout = s, //
                                                .localModified = false,
                                                .hostReadOnly = hostReadOnly,
                                                .remote = RangedAllocation{.ptr = remotePtr, .sizeInBytes = sizeInBytes}});
    remoteToLocalPtr.emplace(remotePtr, RangedAllocation{.ptr = reinterpret_cast<uintptr_t>(local), .sizeInBytes = sizeInBytes});
    return remotePtr;
  }

  void writeSubObject(const char *hostData, const size_t memberOffsetInBytes, const size_t indirection, const uintptr_t devicePtr,
                      const runtime::TypeLayout &tl, const runtime::AggregateMember::ResolvePtrSize resolvePtrSizeInBytes,
                      const auto effectiveSizeToCount) {
    if (debug)
      std::fprintf(stderr,
                   "[SMA] writeSubObject(hostData=%p, memberOffsetInBytes=%zu, indirection=%zu, devicePtr=0x%jx, t=@%s (sizeInBytes=%zd, "
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
      const uintptr_t memberRemotePtr = mirrorToRemote(memberLocalPtr, indirection, effectiveSizeToCount(sizeInBytes), tl, sizeInBytes);
      remoteWrite(&memberRemotePtr, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else if (const PtrQuery meta = localReflect(memberLocalPtr); meta.sizeInBytes != 0) {
      if (debug)
        std::fprintf(stderr, "[SMA]   localReflect(%p) = {size=%ld, offset=%ld}\n", static_cast<void *>(memberLocalPtr), meta.sizeInBytes,
                     meta.offsetInBytes); //
      // XXX Anchor at the allocation base so past-end pointers (offset == size) still map; passing
      // the past-end pointer as host data would request a zero-byte device alloc and fail.
      const bool pastEnd = meta.offsetInBytes == meta.sizeInBytes;
      const char *baseLocalPtr = pastEnd ? memberLocalPtr - meta.offsetInBytes : memberLocalPtr;
      const size_t effectiveSize = pastEnd ? meta.sizeInBytes : meta.sizeInBytes - meta.offsetInBytes;
      const uintptr_t baseRemotePtr =
          mirrorToRemote(baseLocalPtr, indirection, effectiveSizeToCount(effectiveSize), tl, effectiveSize, meta.hostReadOnly);
      const uintptr_t memberRemotePtr = pastEnd ? baseRemotePtr + meta.offsetInBytes : baseRemotePtr;
      remoteWrite(&memberRemotePtr, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else {
      // XXX First-foreign-pointer warning: polyreflect is inactive, so captured heap pointers
      // cannot be sized or mirrored to the device.
      static std::atomic<bool> warnedOnce{false};
      bool expected = false;
      if (warnedOnce.compare_exchange_strong(expected, true)) {
        std::fprintf(
            stderr,
            "[SMA] WARNING: polyreflect tracking is INACTIVE - captured heap pointers will not be mirrored to the device.\n"
            "[SMA]          First foreign pointer %p (type @%s at offset %zu of captures).\n"
            "[SMA]          You might have built with -fstdpar-mem=direct without USM, or the polyreflect plugin failed to load/link.\n"
            "[SMA]          To resolve, rebuild/run with -fstdpar-mem=interpose (clang pass-plugin) or -fstdpar-mem=reflect (LLD "
            "pass-plugin).\n"
            "[SMA]          Further foreign pointers will be silently skipped this run.\n",
            static_cast<void *>(memberLocalPtr), tl.name, memberOffsetInBytes);
      } else if (debug) {
        std::fprintf(stderr, "[SMA] foreign pointer %p when writing type @%s at offset %zu\n", static_cast<void *>(memberLocalPtr), tl.name,
                     memberOffsetInBytes);
      }
    }
  }

  uintptr_t writeIndirect(const char *hostData, const uintptr_t devicePtr, const size_t ptrIndirections, const size_t count,
                          const runtime::TypeLayout &tl) {
    if (debug)
      std::fprintf(stderr, "[SMA] writeIndirect(hostData=%p, count=%ld,devicePtr=0x%jx, t=@ %s, ptrIndirections=%ld)\n", //
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
        } else if (member.type && member.type->memberCount > 0) {
          // member.type is null for synthetic empty-base placeholders (`#base_<Empty>` whose
          // pointee struct has no layout entry in the kernel bundle). They contribute nothing
          // to the marshal — skip rather than dereference null.
          writeStruct(hostData, memberOffsetInBytes, devicePtr, 1, *member.type, false);
        }
      }
    }
    return devicePtr;
  }

  // `hostAllocSizeInBytes` is the actual host allocation size (from localReflect or a ptr-size
  // resolver), or 0 to infer from the type. The two diverge for inheritance-via-base-pointer
  // patterns like std::list: the kernel walks `_List_node_base*` (16 bytes) but the heap node
  // is `_List_node<T>` (24+ bytes); sizing the device alloc to the polyc type would put the
  // payload bytes out-of-bounds.
  uintptr_t mirrorToRemote(const char *hostData, const size_t ptrIndirections, const size_t count, const runtime::TypeLayout &l,
                           const size_t hostAllocSizeInBytes = 0, const bool hostReadOnly = false) {
    if (debug)
      std::fprintf(stderr, "[SMA] mirrorToRemote(hostData=%p, ptrIndirections=%zu, count=%zu, t=@%s, hostAllocSize=%zu)\n", //
                   static_cast<const void *>(hostData), ptrIndirections, count, l.name, hostAllocSizeInBytes);
    if (const auto query = offsetQuery(localToRemoteAlloc, hostData, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      if (!alloc->localModified) {
        if (debug) std::fprintf(stderr, "[SMA]   hit (0x%jx = %4ld + %4ld)\n", alloc->remote.ptr, alloc->remote.sizeInBytes, offsetInBytes);
        return alloc->remote.ptr + offsetInBytes;
      }
      return ptrIndirections > 1 ? writeIndirect(hostData, alloc->remote.ptr, ptrIndirections, count, *alloc->layout)
                                 : writeStruct(hostData, 0, alloc->remote.ptr, count, *alloc->layout, true);
    }
    if (ptrIndirections > 1) {
      return writeIndirect(hostData, createDeviceAllocation(hostData, sizeof(uintptr_t) * count, nullptr), ptrIndirections, count, l);
    }
    const size_t typeSizeTotal = l.sizeInBytes * count;
    const size_t allocSize = std::max(typeSizeTotal, hostAllocSizeInBytes);
    const auto devicePtr = createDeviceAllocation(hostData, allocSize, &l, hostReadOnly);
    // Trailing bytes outside the polyc type (list node payload, vtable slots, etc.) are copied
    // verbatim; writeStruct then walks the known-typed prefix to mirror pointer fields.
    if (allocSize > typeSizeTotal) remoteWrite(hostData + typeSizeTotal, devicePtr, typeSizeTotal, allocSize - typeSizeTotal);
    return writeStruct(hostData, 0, devicePtr, count, l, true);
  }

  void readSubObject(char *p, const size_t memberOffsetInBytes) {
    if (debug)
      std::fprintf(stderr, "[SMA] readSubObject(p=%p, memberOffsetInBytes=%zu)\n", static_cast<const void *>(p), memberOffsetInBytes);
    if (char *memberRemotePtr = readPtrValue(p + memberOffsetInBytes); !memberRemotePtr) {
      std::memset(p + memberOffsetInBytes, 0, sizeof(uintptr_t));
    } else if (const auto query = offsetQuery(
                   remoteToLocalPtr, memberRemotePtr, [](auto &x) { return x.sizeInBytes; },
                   /*allowPastEnd*/ true)) {
      auto [alloc, offset] = *query;
      syncRemoteToLocal(reinterpret_cast<void *>(alloc->ptr), alloc->sizeInBytes);
      // The remote pointer can target an interior position — e.g. the std::list last-node
      // `prev` points back to the sentinel embedded inside the kernel struct. Writing just
      // `alloc->ptr` would clobber those interior references with the allocation base.
      const uintptr_t hostPtr = alloc->ptr + offset;
      std::memcpy(p + memberOffsetInBytes, &hostPtr, sizeof(uintptr_t));
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
      else if (member.type && member.type->memberCount > 0) readStruct(p, offsetInBytes + member.offsetInBytes, member.type);
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
    bool hostReadOnly = false;
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

  std::unordered_set<uintptr_t> syncVisited{};

  std::optional<uintptr_t> syncRemoteToLocal(void *p, const std::optional<size_t> &sizeInByte = {}) { // any valid pointer, not just base
    if (const auto query = offsetQuery(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      // Cycle break for circular structures (a list node's `prev` points to the sentinel
      // embedded inside the kernel struct, which we're already inside the walk of). The
      // top-level call clears the set on its way out so subsequent calls start fresh.
      const bool topLevel = syncVisited.empty();
      if (!syncVisited.insert(alloc->remote.ptr).second) return alloc->remote.ptr + offsetInBytes;
      if (debug)
        std::fprintf(stderr, "[SMA] syncRemoteToLocal(p=%p, remote=0x%jx, sizeInByte=%ld, offsetInBytes=%ld, t=@%s)\n", p,
                     alloc->remote.ptr, alloc->remote.sizeInBytes, offsetInBytes, alloc->layout ? alloc->layout->name : "???");
      // we want to perform an inclusive copy: an object is copied if its range intersects with the request range
      const runtime::TypeLayout *tl = alloc->layout;

      const size_t objSize = tl ? tl->sizeInBytes : sizeof(uintptr_t);
      const size_t objIdxOffset = offsetInBytes % objSize; // offset to return to base at p
      char *baseAtObjIdx = static_cast<char *>(p) - objIdxOffset;

      const size_t maxObjCount = alloc->remote.sizeInBytes / objSize;
      const size_t objIdxBegin = integralFloor(offsetInBytes, objSize);
      const size_t requestedBytes = sizeInByte.value_or(alloc->remote.sizeInBytes - objIdxBegin * objSize);
      const size_t objIdxEnd = std::min(maxObjCount, objIdxBegin + integralCeil(requestedBytes, objSize));
      const size_t totalObjBytes = (objIdxEnd - objIdxBegin) * objSize;

      // XXX skip writeback when host page is RO; cuMemcpyDtoH/hsa fault, and the pointer-rewrite walk would too.
      if (!alloc->hostReadOnly) {
        remoteRead(baseAtObjIdx, alloc->remote.ptr, objIdxBegin * objSize, totalObjBytes);
        if (tl) {
          if (!isSet(tl->attrs, runtime::LayoutAttrs::Opaque)) {
            for (size_t i = objIdxBegin; i < objIdxEnd; ++i)
              readStruct(baseAtObjIdx, i * objSize, tl);
          }
        } else {
          for (size_t i = objIdxBegin; i < objIdxEnd; ++i)
            readSubObject(baseAtObjIdx, i * objSize);
        }
      }
      if (topLevel) syncVisited.clear();
      return alloc->remote.ptr + offsetInBytes;
    }
    return {};
  }

  void disassociate(const void *p, bool releaseRemote = true) { // any valid pointer, not just base
    if (const auto query = offsetQueryIt(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [it, offsetInBytes] = *query;
      if (debug)
        std::fprintf(stderr, "[SMA] disassociate(host=0x%jx, remote=%jx, size=%ld, releaseRemote=%d)\n", it->first, it->second.remote.ptr,
                     it->second.remote.sizeInBytes, releaseRemote);
      if (releaseRemote) remoteRelease(it->second.remote.ptr);
      remoteToLocalPtr.erase(it->second.remote.ptr);
      localToRemoteAlloc.erase(it);
    }
  }
};

} // namespace polyregion::polyrt