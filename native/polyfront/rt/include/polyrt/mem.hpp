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
#include <vector>

#include "polyregion/env_keys.h"
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

  template <typename T> static T readAt(const char *base) {
    T v{};
    std::memcpy(&v, base, sizeof(T));
    return v;
  }
  static char *readPtrValue(const char *base) { return readAt<char *>(base); }

  // XXX `allowPastEnd` matches `localPtr == base + size` (one-past-end). off by default: a forward lookup in
  // `mirrorToRemote` would otherwise attribute an unmapped alloc's base to an adjacent record's past-end
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

  auto queryLocal(const void *p, const bool allowPastEnd = false) {
    return offsetQuery(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; }, allowPastEnd);
  }
  auto queryLocalIt(const void *p, const bool allowPastEnd = false) {
    return offsetQueryIt(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; }, allowPastEnd);
  }
  auto queryRemote(const void *p, const bool allowPastEnd = false) {
    return offsetQuery(remoteToLocalPtr, p, [](auto &x) { return x.sizeInBytes; }, allowPastEnd);
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
                      const size_t pointeeMinBytes, const bool readOnly, const auto effectiveSizeToCount) {
    if (debug)
      std::fprintf(stderr,
                   "[SMA] writeSubObject(hostData=%p, memberOffsetInBytes=%zu, indirection=%zu, devicePtr=0x%jx, t=@%s (sizeInBytes=%zd, "
                   "members=%zd))\n", //
                   static_cast<const void *>(hostData), memberOffsetInBytes, indirection, devicePtr, tl.name, tl.sizeInBytes,
                   tl.memberCount);
    if (char *memberLocalPtr = readPtrValue(hostData + memberOffsetInBytes); !memberLocalPtr) {
      constexpr uintptr_t nullValue = 0;
      remoteWrite(&nullValue, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else if (const size_t resolvedSizeInBytes = resolvePtrSizeInBytes ? resolvePtrSizeInBytes(hostData + memberOffsetInBytes) : 0;
               resolvedSizeInBytes != 0) {
      const size_t sizeInBytes = std::max(resolvedSizeInBytes, pointeeMinBytes);
      if (debug)
        std::fprintf(stderr, "[SMA]   localResolve<%p>(%p) = {size=%ld}\n", //
                     (void *)resolvePtrSizeInBytes, static_cast<void *>(memberLocalPtr), sizeInBytes);
      const uintptr_t memberRemotePtr =
          mirrorToRemote(memberLocalPtr, indirection, effectiveSizeToCount(sizeInBytes), tl, sizeInBytes, readOnly);
      remoteWrite(&memberRemotePtr, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else if (const PtrQuery meta = localReflect(memberLocalPtr); meta.sizeInBytes != 0) {
      if (debug)
        std::fprintf(stderr, "[SMA]   localReflect(%p) = {size=%ld, offset=%ld}\n", static_cast<void *>(memberLocalPtr), meta.sizeInBytes,
                     meta.offsetInBytes); //
      // XXX Anchor at the allocation base so past-end pointers (offset == size) still map; passing
      // the past-end pointer as host data would request a zero-byte device alloc and fail.
      const bool pastEnd = meta.offsetInBytes == meta.sizeInBytes;
      const char *baseLocalPtr = pastEnd ? memberLocalPtr - meta.offsetInBytes : memberLocalPtr;
      const size_t effectiveSize = std::max(pastEnd ? meta.sizeInBytes : meta.sizeInBytes - meta.offsetInBytes, pointeeMinBytes);
      const uintptr_t baseRemotePtr =
          mirrorToRemote(baseLocalPtr, indirection, effectiveSizeToCount(effectiveSize), tl, effectiveSize, readOnly || meta.hostReadOnly);
      const uintptr_t memberRemotePtr = pastEnd ? baseRemotePtr + meta.offsetInBytes : baseRemotePtr;
      remoteWrite(&memberRemotePtr, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else {
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
                          const runtime::TypeLayout &tl, const bool readOnly = false) {
    if (debug)
      std::fprintf(stderr, "[SMA] writeIndirect(hostData=%p, count=%ld,devicePtr=0x%jx, t=@ %s, ptrIndirections=%ld)\n", //
                   static_cast<const void *>(hostData), count, devicePtr, tl.name, ptrIndirections);
    for (size_t idx = 0; idx < count; ++idx) {
      const size_t componentSize = ptrIndirections - 1 == 1 ? tl.sizeInBytes : sizeof(uintptr_t);
      writeSubObject(hostData, idx * sizeof(uintptr_t), ptrIndirections - 1, devicePtr, tl, nullptr, /*pointeeMinBytes=*/0, readOnly,
                     [&](const size_t effectiveSize) { return effectiveSize / componentSize; });
    }
    return devicePtr;
  }

  uintptr_t writeStruct(const char *hostData, const size_t offsetInBytes, const uintptr_t devicePtr, const size_t count,
                        const runtime::TypeLayout &tl, const bool writeBody, const bool readOnly = false) {
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
        const bool memberReadOnly = readOnly || member.readOnly != 0;
        if (member.ptrIndirection > 0) {
          const size_t componentSize = member.ptrIndirection == 1 ? member.componentSize : member.sizeInBytes;
          const size_t pointeeMinBytes = member.ptrIndirection == 1 ? member.componentSize : 0;
          writeSubObject(hostData, memberOffsetInBytes, member.ptrIndirection, devicePtr, *member.type, member.resolvePtrSizeInBytes,
                         pointeeMinBytes, memberReadOnly, [&](const size_t effectiveSize) { return effectiveSize / componentSize; });
        } else if (member.type && member.type->memberCount > 0) {
          // member.type is null for synthetic empty-base placeholders (`#base_<Empty>`); skip, don't deref
          writeStruct(hostData, memberOffsetInBytes, devicePtr, 1, *member.type, false, memberReadOnly);
        }
      }
    }
    return devicePtr;
  }

  // `hostAllocSizeInBytes` is the real host allocation size (from localReflect / a ptr-size resolver), or 0 to
  // infer from the type. they diverge for inheritance-via-base-pointer (std::list): the kernel walks
  // `_List_node_base*` (16B) but the heap node is `_List_node<T>` (24B+); sizing to the polyc type OOBs the payload
  uintptr_t mirrorToRemote(const char *hostData, const size_t ptrIndirections, const size_t count, const runtime::TypeLayout &l,
                           const size_t hostAllocSizeInBytes = 0, const bool hostReadOnly = false) {
    if (debug)
      std::fprintf(stderr, "[SMA] mirrorToRemote(hostData=%p, ptrIndirections=%zu, count=%zu, t=@%s, hostAllocSize=%zu)\n", //
                   static_cast<const void *>(hostData), ptrIndirections, count, l.name, hostAllocSizeInBytes);
    if (const auto query = queryLocal(hostData)) {
      const auto [alloc, offsetInBytes] = *query;
      const bool eligible = ptrIndirections == 1 && offsetInBytes == 0;
      const size_t cachedSize = alloc->remote.sizeInBytes;
      const size_t requiredSize = staleFloor(l.sizeInBytes * count, hostAllocSizeInBytes);
      const bool stale = eligible && cachedSize < requiredSize;
      if (stale) {
        disassociate(hostData);
        if (debug) std::fprintf(stderr, "[SMA]   stale base hit (cached %ld < required %ld), re-mirroring\n", cachedSize, requiredSize);
      } else if (!alloc->localModified) {
        if (debug) std::fprintf(stderr, "[SMA]   hit (0x%jx = %4ld + %4ld)\n", alloc->remote.ptr, alloc->remote.sizeInBytes, offsetInBytes);
        return alloc->remote.ptr + offsetInBytes;
      } else {
        const auto &effLayout = alloc->layout ? *alloc->layout : l;
        const bool ro = hostReadOnly || alloc->hostReadOnly;
        return ptrIndirections > 1 ? writeIndirect(hostData, alloc->remote.ptr, ptrIndirections, count, effLayout, ro)
                                   : writeStruct(hostData, 0, alloc->remote.ptr, count, effLayout, true, ro);
      }
    }
    if (ptrIndirections > 1) {
      return writeIndirect(hostData, createDeviceAllocation(hostData, sizeof(uintptr_t) * count, nullptr, hostReadOnly), ptrIndirections,
                           count, l, hostReadOnly);
    }
    const size_t typeSizeTotal = l.sizeInBytes * count;
    const size_t allocSize = std::max(typeSizeTotal, hostAllocSizeInBytes);
    const auto devicePtr = createDeviceAllocation(hostData, allocSize, &l, hostReadOnly);
    // Trailing bytes outside the polyc type (list node payload, vtable slots, etc.) are copied
    // verbatim; writeStruct then walks the known-typed prefix to mirror pointer fields.
    if (allocSize > typeSizeTotal) remoteWrite(hostData + typeSizeTotal, devicePtr, typeSizeTotal, allocSize - typeSizeTotal);
    return writeStruct(hostData, 0, devicePtr, count, l, true, hostReadOnly);
  }

  // orig = host bytes snapshotted before the read-back clobber, to restore a slot left foreign
  void readSubObject(char *p, const size_t memberOffsetInBytes, const char *orig, const size_t origLen) {
    if (debug)
      std::fprintf(stderr, "[SMA] readSubObject(p=%p, memberOffsetInBytes=%zu)\n", static_cast<const void *>(p), memberOffsetInBytes);
    if (char *memberRemotePtr = readPtrValue(p + memberOffsetInBytes); !memberRemotePtr) {
      std::memset(p + memberOffsetInBytes, 0, sizeof(uintptr_t));
    } else if (const auto query = queryRemote(memberRemotePtr, /*allowPastEnd*/ true)) {
      auto [alloc, offset] = *query;
      syncRemoteToLocal(reinterpret_cast<void *>(alloc->ptr), alloc->sizeInBytes);
      // remote ptr can be interior (std::list last-node prev -> embedded sentinel); add offset, don't snap to base
      const uintptr_t hostPtr = alloc->ptr + offset;
      std::memcpy(p + memberOffsetInBytes, &hostPtr, sizeof(uintptr_t));
    } else if (const char *origVal =
                   (orig && memberOffsetInBytes + sizeof(uintptr_t) <= origLen) ? readPtrValue(orig + memberOffsetInBytes) : nullptr;
               origVal && (queryLocal(origVal) || localReflect(origVal).sizeInBytes != 0)) {
      // read-back left a stale device address where the host had a live pointer; restore it, don't plant a wild ptr
      std::memcpy(p + memberOffsetInBytes, &origVal, sizeof(uintptr_t));
    } else if (debug) {
      std::fprintf(stderr, "[SMA] foreign member pointer %p at offset %zu\n", static_cast<void *>(memberRemotePtr), memberOffsetInBytes);
    }
  }

  void readStruct(char *p, const size_t offsetInBytes, const runtime::TypeLayout *tl, const char *orig, const size_t origLen) {
    if (debug)
      std::fprintf(stderr, "[SMA] readStruct(p=%p, offsetInBytes=%zu, t=@%s)\n", static_cast<const void *>(p), offsetInBytes, tl->name);
    for (size_t m = 0; m < tl->memberCount; ++m) {
      const auto member = tl->members[m];
      if (member.sizeInBytes == 0) continue;
      if (member.ptrIndirection > 0) readSubObject(p, offsetInBytes + member.offsetInBytes, orig, origLen);
      else if (member.type && member.type->memberCount > 0) readStruct(p, offsetInBytes + member.offsetInBytes, member.type, orig, origLen);
    }
  }

  // gather host pointer values of `tl`'s pointer members (recursing embedded structs), read BEFORE the device
  // read-back clobbers them so syncRemoteToLocal can still reach nodes a kernel disconnected from the root
  void collectHostPtrs(const char *base, const size_t offsetInBytes, const runtime::TypeLayout *tl, std::vector<void *> &out) {
    for (size_t m = 0; m < tl->memberCount; ++m) {
      const auto member = tl->members[m];
      if (member.sizeInBytes == 0) continue;
      if (member.ptrIndirection > 0) {
        if (const auto ptr = readPtrValue(base + offsetInBytes + member.offsetInBytes)) out.push_back(const_cast<char *>(ptr));
      } else if (member.type && member.type->memberCount > 0) collectHostPtrs(base, offsetInBytes + member.offsetInBytes, member.type, out);
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
    if (const auto query = queryLocal(p)) {
      const auto [alloc, offsetInBytes] = *query;
      if (!alloc->localModified) return alloc->remote.ptr + offsetInBytes;
      else
        return writeStruct(static_cast<const char *>(p), 0, alloc->remote.ptr, alloc->remote.sizeInBytes / alloc->layout->sizeInBytes,
                           *alloc->layout, true, alloc->hostReadOnly);
    }
    return {};
  }

  std::optional<uintptr_t> invalidateLocal(const void *p) {
    if (debug) std::fprintf(stderr, "[SMA] invalidateLocal(p=%p)\n", p);
    if (const auto query = queryLocal(p)) {
      const auto [alloc, offsetInBytes] = *query;
      alloc->localModified = true;
      return alloc->remote.ptr + offsetInBytes;
    }
    return {};
  }

  std::unordered_set<uintptr_t> syncVisited{};

  std::optional<uintptr_t> syncRemoteToLocal(void *p, const std::optional<size_t> &sizeInByte = {}) {
    if (const auto query = queryLocal(p)) {
      const auto [alloc, offsetInBytes] = *query;
      // cycle break for circular structures (a list node's `prev` -> the sentinel embedded in the kernel
      // struct we are already walking); the top-level call clears the set on exit so later calls start fresh
      const bool topLevel = syncVisited.empty();
      if (!syncVisited.insert(alloc->remote.ptr).second) return alloc->remote.ptr + offsetInBytes;
      if (debug)
        std::fprintf(stderr, "[SMA] syncRemoteToLocal(p=%p, remote=0x%jx, sizeInByte=%ld, offsetInBytes=%ld, t=@%s)\n", p,
                     alloc->remote.ptr, alloc->remote.sizeInBytes, offsetInBytes, alloc->layout ? alloc->layout->name : "???");
      // inclusive copy: an object is copied if its range intersects the request range
      const runtime::TypeLayout *tl = alloc->layout;

      const size_t objSize = tl ? tl->sizeInBytes : sizeof(uintptr_t);
      const size_t objIdxOffset = offsetInBytes % objSize;
      char *baseAtObjIdx = static_cast<char *>(p) - objIdxOffset;

      const size_t maxObjCount = alloc->remote.sizeInBytes / objSize;
      const size_t objIdxBegin = integralFloor(offsetInBytes, objSize);
      const size_t startByte = objIdxBegin * objSize;
      const size_t requestedBytes = sizeInByte.value_or(alloc->remote.sizeInBytes - startByte);
      const size_t objIdxEnd = std::min(maxObjCount, objIdxBegin + integralCeil(requestedBytes, objSize));
      const size_t totalObjBytes =
          std::max(objIdxEnd * objSize, std::min(alloc->remote.sizeInBytes, startByte + requestedBytes)) - startByte;

      // XXX skip writeback when host page is RO; cuMemcpyDtoH/hsa fault, and the pointer-rewrite walk would too.
      if (!alloc->hostReadOnly) {
        const bool walkable = tl ? !isSet(tl->attrs, runtime::LayoutAttrs::Opaque) : true;
        // snapshot the original outgoing pointers before the read-back overwrites them with device addresses
        std::vector<void *> origPtrs;
        // snapshot for readSubObject to restore a slot the read-back leaves untranslatable
        std::vector<char> origBytes;
        // index from baseAtObjIdx (= alloc byte startByte): local (i - objIdxBegin) * objSize, not global i * objSize
        if (walkable) {
          origBytes.assign(baseAtObjIdx, baseAtObjIdx + totalObjBytes);
          for (size_t i = objIdxBegin; i < objIdxEnd; ++i) {
            const size_t localOff = (i - objIdxBegin) * objSize;
            if (tl) collectHostPtrs(baseAtObjIdx, localOff, tl, origPtrs);
            else if (const auto ptr = readPtrValue(baseAtObjIdx + localOff)) origPtrs.push_back(const_cast<char *>(ptr));
          }
        }
        remoteRead(baseAtObjIdx, alloc->remote.ptr, startByte, totalObjBytes);
        if (walkable)
          for (size_t i = objIdxBegin; i < objIdxEnd; ++i) {
            const size_t localOff = (i - objIdxBegin) * objSize;
            if (tl) readStruct(baseAtObjIdx, localOff, tl, origBytes.data(), totalObjBytes);
            else readSubObject(baseAtObjIdx, localOff, origBytes.data(), totalObjBytes);
          }
        // follow the original graph too: a kernel that disconnects nodes (reverses a list) leaves them
        // unreachable through the read-back device pointers, but they are still mirrored and must come home
        for (void *op : origPtrs)
          syncRemoteToLocal(op);
      }
      if (topLevel) syncVisited.clear();
      return alloc->remote.ptr + offsetInBytes;
    }
    return {};
  }

  void disassociate(const void *p, bool releaseRemote = true) {
    if (const auto query = queryLocalIt(p)) {
      const auto [it, offsetInBytes] = *query;
      if (debug)
        std::fprintf(stderr, "[SMA] disassociate(host=0x%jx, remote=%jx, size=%ld, releaseRemote=%d)\n", it->first, it->second.remote.ptr,
                     it->second.remote.sizeInBytes, releaseRemote);
      if (releaseRemote) remoteRelease(it->second.remote.ptr);
      remoteToLocalPtr.erase(it->second.remote.ptr);
      localToRemoteAlloc.erase(it);
    }
  }

  void disassociateExact(const void *p, const bool releaseRemote = true) {
    const auto it = localToRemoteAlloc.find(reinterpret_cast<uintptr_t>(p));
    if (it == localToRemoteAlloc.end()) return;
    if (debug)
      std::fprintf(stderr, "[SMA] disassociateExact(host=0x%jx, remote=%jx, size=%ld)\n", it->first, it->second.remote.ptr,
                   it->second.remote.sizeInBytes);
    if (releaseRemote) remoteRelease(it->second.remote.ptr);
    remoteToLocalPtr.erase(it->second.remote.ptr);
    localToRemoteAlloc.erase(it);
  }

  uintptr_t mirrorAlloc(const void *local, const size_t sizeInBytes, const bool hostReadOnly) {
    const uintptr_t remote = createDeviceAllocation(static_cast<const char *>(local), sizeInBytes, nullptr, hostReadOnly);
    remoteWrite(local, remote, 0, sizeInBytes);
    return remote;
  }

  bool deviceWriteInBounds(const uintptr_t remote, const size_t offsetInBytes, const size_t size) {
    if (const auto q = queryRemote(reinterpret_cast<const void *>(remote), /*allowPastEnd*/ true))
      return q->second + offsetInBytes + size <= q->first->sizeInBytes;
    return false;
  }

  void mirrorPatch(const uintptr_t remote, const size_t offsetInBytes, const uintptr_t value) {
    // value==0 is a foreign pointee already at a valid device address; leave the copied original
    if (value == 0) return;
    if (deviceWriteInBounds(remote, offsetInBytes, sizeof(uintptr_t))) remoteWrite(&value, remote, offsetInBytes, sizeof(uintptr_t));
  }

  template <typename Q> bool pastEndAliasOfDistinct(const void *local, const Q &query) {
    if (query->second != query->first->remote.sizeInBytes) return false;
    const PtrQuery m = localReflect(local);
    return m.sizeInBytes != 0 && m.offsetInBytes == 0;
  }

  static size_t staleFloor(size_t reflectedSize, size_t minBytes) { return std::max(reflectedSize, minBytes); }

  std::optional<std::pair<uintptr_t, size_t>> ensureMirroredBase(const void *local, size_t minSize) {
    const PtrQuery meta = localReflect(local);
    const auto query = queryLocal(local, /*allowPastEnd*/ true);
    const bool cacheHit = query && !pastEndAliasOfDistinct(local, query);
    const size_t sz = meta.sizeInBytes == 0 ? 0 : staleFloor(meta.sizeInBytes, minSize);
    if (cacheHit) {
      if (sz <= query->first->remote.sizeInBytes) return std::pair{query->first->remote.ptr, query->second};
      disassociate(local);
    }
    if (sz == 0) return std::nullopt;
    const size_t off = meta.sizeInBytes == 0 ? 0 : meta.offsetInBytes;
    const char *base = static_cast<const char *>(local) - off;
    const uintptr_t baseRemote = createDeviceAllocation(base, sz, nullptr, meta.hostReadOnly);
    remoteWrite(base, baseRemote, 0, sz);
    return std::pair{baseRemote, off};
  }

  uintptr_t mirrorEnsure(const void *local, size_t minSize = 0) {
    const auto r = ensureMirroredBase(local, minSize);
    return r ? r->first + r->second : 0;
  }

  size_t mirrorPointeeSize(const void *local) {
    if (!local) return 0;
    const PtrQuery meta = localReflect(local);
    return meta.sizeInBytes > meta.offsetInBytes ? meta.sizeInBytes - meta.offsetInBytes : meta.sizeInBytes;
  }

  uintptr_t mirrorEnsureDeep(const void *local, const size_t depth) {
    if (debug) std::fprintf(stderr, "[SMA] mirrorEnsureDeep(local=%p, depth=%zu)\n", local, depth);
    const uintptr_t devBlock = mirrorEnsure(local);
    if (devBlock == 0 || depth == 0) return devBlock;
    // bound the walk by the mirrored device alloc, not a fresh localReflect, else writing past it faults
    const auto query = queryLocal(local, /*allowPastEnd*/ true);
    if (!query) return devBlock;
    const auto [alloc, offset] = *query;
    const size_t allocSize = alloc->remote.sizeInBytes;
    const char *base = static_cast<const char *>(local) - offset;
    const uintptr_t devBase = alloc->remote.ptr;
    const auto follow = [&](const size_t byteOff) {
      if (byteOff + sizeof(uintptr_t) > allocSize) return;
      const auto elem = readPtrValue(base + byteOff);
      if (!elem) return;
      // an interior leaf pointee is fine (a .rodata static array at a segment offset), but a deeper level needs a base
      if (depth > 1 && localReflect(elem).offsetInBytes != 0) return;
      if (const uintptr_t devElem = mirrorEnsureDeep(elem, depth - 1)) mirrorPatch(devBase, byteOff, devElem);
    };
    if (offset == 0)
      for (size_t o = 0; o + sizeof(uintptr_t) <= allocSize; o += sizeof(uintptr_t))
        follow(o);
    else follow(offset);
    return devBlock;
  }

  void unmirrorReadAlloc(const void *local) {
    if (const auto query = queryLocal(local, /*allowPastEnd*/ true)) {
      const auto [alloc, offset] = *query;
      if (alloc->hostReadOnly) return;
      if (!syncVisited.insert(alloc->remote.ptr).second) return;
      // interior into a structured alloc (an SSO string's _M_p) is restored by the object's own read-back;
      // raw data (null layout) has none, and MSVC over-aligns big std::vector so its data pointer is interior
      if (offset != 0 && alloc->layout) return;
      const char *base = static_cast<const char *>(local) - offset;
      remoteRead(const_cast<char *>(base), alloc->remote.ptr, 0, alloc->remote.sizeInBytes);
    }
  }

  // dual of mirrorEnsureDeep: at each level translate a pointer slot the kernel changed back to its host
  // address (like readSubObject), then follow it to read deeper; a slot the kernel left alone is not rewritten
  void unmirrorReadDeep(const void *local, const size_t depth) {
    if (debug) std::fprintf(stderr, "[SMA] unmirrorReadDeep(local=%p, depth=%zu)\n", local, depth);
    if (depth == 0) {
      unmirrorReadAlloc(local);
      return;
    }
    const auto query = queryLocal(local, /*allowPastEnd*/ true);
    if (!query) return;
    const auto [alloc, offset] = *query;
    const size_t allocSize = alloc->remote.sizeInBytes;
    const bool ro = alloc->hostReadOnly;
    const uintptr_t devBase = alloc->remote.ptr;
    char *base = const_cast<char *>(static_cast<const char *>(local)) - offset;
    std::vector<char> buf(allocSize);
    remoteRead(buf.data(), devBase, 0, allocSize);
    const auto step = [&](const size_t byteOff) {
      if (byteOff + sizeof(uintptr_t) > allocSize) return;
      if (!ro) {
        const uintptr_t devVal = readAt<uintptr_t>(buf.data() + byteOff);
        const char *cur = readPtrValue(base + byteOff);
        if (devVal == 0) {
          if (cur) std::memset(base + byteOff, 0, sizeof(uintptr_t));
        } else if (const auto q = queryRemote(reinterpret_cast<void *>(devVal), /*allowPastEnd*/ true)) {
          if (const char *hostNew = reinterpret_cast<const char *>(q->first->ptr + q->second); hostNew != cur)
            std::memcpy(base + byteOff, &hostNew, sizeof(uintptr_t));
        }
      }
      if (const char *elem = readPtrValue(base + byteOff)) {
        if (depth > 1 && localReflect(elem).offsetInBytes != 0) return; // interior leaf ok, a deeper level needs a base
        unmirrorReadDeep(elem, depth - 1);
      }
    };
    if (offset == 0)
      for (size_t o = 0; o + sizeof(uintptr_t) <= allocSize; o += sizeof(uintptr_t))
        step(o);
    else step(offset);
  }

  void unmirrorVisitClear() { syncVisited.clear(); }

  struct MirroredGraph {
    std::vector<size_t> ptrOffsets; // sorted
    std::vector<const char *> nodes;
    std::vector<uintptr_t> devs;
    std::vector<std::vector<uintptr_t>> origPtrs; // [node][ptrOffsets idx] device value written at mirror time
  };
  std::unordered_map<const void *, MirroredGraph> mirroredGraphs{};

  // generic single-arena offset-rewrite mirror: the binding-slot dual of mirrorToRemote. the whole reachable
  // graph is laid into one host byte arena, every pointer slot rewritten to a byte offset into it (null ->
  // arenaNull); genArenaFinish uploads it once. cycles/sharing dedup by host base, so depth is unbounded.
  // null sentinel = 0: the capture root sits at arena offset 0 with nothing pointing back, so real offsets are > 0
  static constexpr uint64_t arenaNull = 0;
  std::vector<char> genArenaStaging{};
  std::unordered_map<uintptr_t, uint64_t> genArenaOffsetOf{};
  uintptr_t genArenaDevBase = 0;
  size_t genArenaDevCapacity = 0; // reuse the device arena across launches; realloc only to grow
  size_t genArenaDevValid = 0;    // persist: bytes of live device content to carry across a realloc (kernel outputs)
  bool genArenaActive = false;    // a backend may hand out device handle 0 (cl_mem index), so track validity separately
  // trailing zeroed pad after each arena object: the arena packs objects adjacently, so a vectoriser's
  // over-read past an array end must land in zeroed slack, not neighbour bytes (Mesa/llvmpipe runs
  // unpredicated SIMD-remainder lanes). set from the device's overread_pad feature (0 = off for sound backends)
  size_t genArenaObjectSlack = 0;
  // every mirrored object, for the read-back walk (dual of genArenaMirror)
  struct GenArenaRec {
    const char *host;
    uint64_t off;
    size_t count, indirection, size;
    const runtime::TypeLayout *l;
    bool readOnly;            // a const input the device cannot have written; never copy it back (would corrupt it)
    bool hostDirty = false;   // persist mode: host wrote this leaf since the last upload, so re-send it
    bool deviceDirty = false; // persist mode: a kernel wrote this leaf; the host copy is stale until map_read
  };
  std::vector<GenArenaRec> genArenaRecs{};

  // persist mode: captured leaves stay resident across launches at stable offsets so only the capture root
  // (re-laid each launch) and host-dirtied/new leaves re-upload, and a kernel's output flows to the next
  // kernel on-device. the capture sits in a reserved [0, CAP) prefix (so the kernel still reads root at 0 and
  // leaf offsets never shift); a bigger root grows CAP and re-lays the union. correct for deep graphs too -
  // a leaf's inner pointer holds the pointee's absolute (stable) offset
  bool genArenaPersist = std::getenv(env::PolyrtArenaPersist) != nullptr;
  bool genArenaSeenRoot = false;                              // first mirror call of the launch is the capture root
  size_t genArenaCaptureCap = 0;                              // reserved capture-region size; leaves bump-allocate above it
  std::vector<std::pair<uint64_t, size_t>> genArenaUploads{}; // regions to H2D this launch (capture + dirty/new leaves)
  std::vector<uint64_t> genArenaTouched{};                    // leaf offsets this kernel reached, marked device-dirty after the launch

  void genArenaReset() {
    genArenaUploads.clear();
    genArenaTouched.clear();
    genArenaSeenRoot = false;
    if (genArenaPersist && genArenaCaptureCap) {
      // keep resident leaves; drop only the transient root rec (off 0)
      genArenaRecs.erase(std::remove_if(genArenaRecs.begin(), genArenaRecs.end(), [](auto &r) { return r.off == 0; }), genArenaRecs.end());
      return;
    }
    genArenaStaging.clear();
    genArenaOffsetOf.clear();
    genArenaRecs.clear();
  }

  // find the resident leaf a host pointer falls in; drives map_read/map_write dirty tracking in persist mode
  GenArenaRec *genArenaRecOf(const void *p) {
    const auto a = reinterpret_cast<const char *>(p);
    for (auto &r : genArenaRecs)
      if (r.off != 0 && a >= r.host && a < r.host + r.size) return &r;
    return nullptr;
  }

  // patch the pointer at host `hostBase+hostOff` (already copied into the arena at `arenaPtrOff`) to its
  // pointee's arena offset, recursing the pointee
  void genArenaPatch(const char *hostBase, const size_t hostOff, const uint64_t arenaPtrOff, const size_t indirection,
                     const size_t componentSize, const size_t pointeeMinBytes, const runtime::TypeLayout &tl,
                     const runtime::AggregateMember::ResolvePtrSize resolve, const bool readOnly) {
    uint64_t childOff = arenaNull;
    if (const char *target = readPtrValue(hostBase + hostOff); target) {
      // interior pointer into an already-mirrored object (e.g. an SSO string's `_M_p` -> its own
      // `_M_local_buf`): point at that object's interior, don't re-mirror or fall to arenaNull
      for (const auto &r : genArenaRecs)
        if (target >= r.host && target < r.host + r.size) {
          childOff = r.off + (target - r.host);
          break;
        }
      if (childOff != arenaNull) {
        std::memcpy(genArenaStaging.data() + arenaPtrOff, &childOff, sizeof(uint64_t));
        return;
      }
      const char *base = target;
      size_t sz = 0, pastEndAdd = 0;
      bool sized = false, ro = readOnly;
      if (resolve) {
        if (const size_t resolvedSize = resolve(hostBase + hostOff); resolvedSize != 0) {
          sz = std::max(resolvedSize, pointeeMinBytes);
          sized = true;
        }
      }
      if (!sized) {
        if (const PtrQuery meta = localReflect(target); meta.sizeInBytes != 0) {
          const bool pastEnd = meta.offsetInBytes == meta.sizeInBytes;
          base = pastEnd ? target - meta.offsetInBytes : target;
          sz = std::max(pastEnd ? meta.sizeInBytes : meta.sizeInBytes - meta.offsetInBytes, pointeeMinBytes);
          pastEndAdd = pastEnd ? meta.offsetInBytes : 0;
          ro = ro || meta.hostReadOnly;
          sized = true;
        }
      }
      if (sized) {
        const size_t cnt = componentSize ? sz / componentSize : 1;
        childOff = genArenaMirror(base, cnt ? cnt : 1, indirection, tl, sz, ro);
        if (childOff != arenaNull) childOff += pastEndAdd;
      } else return; // XXX foreign / unsized: preserve copied bytes
    }
    std::memcpy(genArenaStaging.data() + arenaPtrOff, &childOff, sizeof(uint64_t));
  }

  // recurse the typed prefix copied at arena `arenaOff` (mirror of host `hostBase+hostOff`); a const link
  // anywhere on the path makes the pointee read-only (the device cannot write through it)
  void genArenaWalk(const char *hostBase, const uint64_t arenaOff, const size_t hostOff, const size_t count, const runtime::TypeLayout &l,
                    const bool readOnly) {
    for (size_t idx = 0; idx < count; ++idx)
      for (size_t m = 0; m < l.memberCount; ++m) {
        const auto member = l.members[m];
        if (member.sizeInBytes == 0) continue;
        const size_t hOff = hostOff + idx * l.sizeInBytes + member.offsetInBytes;
        const uint64_t aOff = arenaOff + idx * l.sizeInBytes + member.offsetInBytes;
        const bool memberRO = readOnly || member.readOnly != 0;
        if (member.ptrIndirection > 0) {
          const size_t comp = member.ptrIndirection == 1 ? member.componentSize : sizeof(uintptr_t);
          const size_t minB = member.ptrIndirection == 1 ? member.componentSize : 0;
          genArenaPatch(hostBase, hOff, aOff, member.ptrIndirection, comp, minB, *member.type, member.resolvePtrSizeInBytes, memberRO);
        } else if (member.type && member.type->memberCount > 0) genArenaWalk(hostBase, aOff, hOff, 1, *member.type, memberRO);
      }
  }

  // append `count` elements of `hostData` to the arena and patch their pointers; dedup by host ptr.
  // indirection>1: the elements are themselves pointers (an array of pointers)
  uint64_t genArenaMirror(const char *hostData, const size_t count, const size_t indirection, const runtime::TypeLayout &l,
                          const size_t hostAllocSizeInBytes = 0, const bool readOnly = false) {
    if (!hostData) return arenaNull;
    const size_t unit = indirection > 1 ? sizeof(uintptr_t) : l.sizeInBytes;
    const size_t allocSize = std::max(unit * count, hostAllocSizeInBytes);

    if (genArenaPersist) {
      if (!genArenaSeenRoot) { // the capture root: re-laid at offset 0 each launch, never persisted/deduped
        genArenaSeenRoot = true;
        if (allocSize > genArenaCaptureCap) { // grow the reserved prefix and re-lay the whole union below it
          genArenaCaptureCap = (allocSize + 255) & ~size_t(255);
          genArenaStaging.assign(genArenaCaptureCap, 0);
          genArenaOffsetOf.clear();
          genArenaRecs.clear();
          genArenaDevValid = 0; // layout changed: the resident device image is stale, force a full re-upload
        } else if (genArenaStaging.size() < genArenaCaptureCap) genArenaStaging.assign(genArenaCaptureCap, 0);
        std::memcpy(genArenaStaging.data(), hostData, allocSize);
        std::memset(genArenaStaging.data() + allocSize, 0, genArenaCaptureCap - allocSize); // zero slack for over-reads
        genArenaRecs.push_back({hostData, 0, count, indirection, allocSize, &l, readOnly});
        genArenaUploads.emplace_back(0, genArenaCaptureCap);
        if (indirection > 1)
          for (size_t i = 0; i < count; ++i)
            genArenaPatch(hostData, i * sizeof(uintptr_t), i * sizeof(uintptr_t), indirection - 1,
                          indirection - 1 == 1 ? l.sizeInBytes : sizeof(uintptr_t), 0, l, nullptr, readOnly);
        else genArenaWalk(hostData, 0, 0, count, l, readOnly);
        return 0;
      }
      // a resident leaf: the device copy is current (kernel-to-kernel flow); re-send only if the host dirtied it
      if (const auto it = genArenaOffsetOf.find(reinterpret_cast<uintptr_t>(hostData)); it != genArenaOffsetOf.end()) {
        const uint64_t off = it->second;
        genArenaTouched.push_back(off);
        for (auto &r : genArenaRecs)
          if (r.off == off && r.hostDirty) {
            std::memcpy(genArenaStaging.data() + off, hostData, r.size);
            genArenaUploads.emplace_back(off, r.size);
            r.hostDirty = false;
          }
        return off;
      }
      // new leaf: fall through to bump-allocate above the capture region, then schedule its upload + touch
    } else if (const auto it = genArenaOffsetOf.find(reinterpret_cast<uintptr_t>(hostData)); it != genArenaOffsetOf.end())
      return it->second;

    // pad to the object's alignment: the device reads each pointer/scalar field via a typed access-chain
    // off the arena base, so an under-aligned offset faults (CUDA misaligned-address / SPIR-V UB)
    const size_t align = indirection > 1 ? sizeof(uintptr_t) : std::max<size_t>(1, l.alignmentInBytes);
    if (const size_t pad = (align - genArenaStaging.size() % align) % align; pad) genArenaStaging.resize(genArenaStaging.size() + pad);
    const uint64_t off = genArenaStaging.size();
    genArenaOffsetOf[reinterpret_cast<uintptr_t>(hostData)] = off;
    genArenaRecs.push_back({hostData, off, count, indirection, allocSize, &l, readOnly});
    genArenaStaging.insert(genArenaStaging.end(), hostData, hostData + allocSize);
    // pad before any child object is appended, so this object's over-read lands in its own zeroed slack
    if (genArenaObjectSlack) genArenaStaging.resize(genArenaStaging.size() + genArenaObjectSlack);
    if (genArenaPersist) {
      genArenaUploads.emplace_back(off, allocSize);
      genArenaTouched.push_back(off);
    }
    if (indirection > 1)
      for (size_t i = 0; i < count; ++i)
        genArenaPatch(hostData, i * sizeof(uintptr_t), off + i * sizeof(uintptr_t), indirection - 1,
                      indirection - 1 == 1 ? l.sizeInBytes : sizeof(uintptr_t), 0, l, nullptr, readOnly);
    else genArenaWalk(hostData, off, 0, count, l, readOnly);
    return off;
  }

  uintptr_t genArenaFinish() {
    const size_t need = genArenaStaging.size() + 1;
    if (!genArenaActive || need > genArenaDevCapacity) {
      // the resident leaves carry kernel outputs (not in staging), so preserve the live device image across
      // the realloc instead of re-uploading stale staging - else a growth (a later kernel's new leaf) clobbers
      // every prior kernel's result. headroom keeps growths logarithmic during warmup
      std::vector<char> keep;
      if (genArenaPersist && genArenaActive && genArenaDevValid) {
        keep.resize(genArenaDevValid);
        remoteRead(keep.data(), genArenaDevBase, 0, genArenaDevValid);
      }
      if (genArenaActive) remoteRelease(genArenaDevBase);
      genArenaDevCapacity = std::max(need, genArenaDevCapacity * 2);
      genArenaDevBase = remoteAlloc(genArenaDevCapacity);
      genArenaActive = true;
      if (!keep.empty()) remoteWrite(keep.data(), genArenaDevBase, 0, keep.size());
    }
    if (genArenaPersist && genArenaDevValid) {
      // resident leaves are already on-device (just restored if we grew); only the capture + dirty/new leaves
      for (const auto &[off, sz] : genArenaUploads)
        remoteWrite(genArenaStaging.data() + off, genArenaDevBase, off, sz);
      if (debug)
        std::fprintf(stderr, "[SMA] genArenaFinish(persist) -> %zu region(s), %zu bytes resident\n", genArenaUploads.size(),
                     genArenaStaging.size());
    } else {
      // first launch (or a re-laid union): nothing resident yet, upload the whole image once
      remoteWrite(genArenaStaging.data(), genArenaDevBase, 0, genArenaStaging.size());
      if (debug) std::fprintf(stderr, "[SMA] genArenaFinish -> %zu bytes\n", genArenaStaging.size());
    }
    genArenaDevValid = genArenaStaging.size();
    return genArenaDevBase;
  }

  // inverse of genArenaPatch's interior search: a device arena byte offset -> the host address it mirrors.
  // arenaNull -> nullptr; inside a rec -> its host interior; in no rec (foreign / seam) -> nullopt (leave untouched)
  std::optional<const char *> genArenaHostOf(const uint64_t off) const {
    if (off == arenaNull) return static_cast<const char *>(nullptr);
    for (const auto &r : genArenaRecs)
      if (off >= r.off && off < r.off + r.size) return r.host + (off - r.off);
    return std::nullopt;
  }

  // translate one device pointer slot (arena offset at `aOff`) back to the host pointer at `hostSlot`, but only
  // when the kernel changed it: a slot left as arenaNull (foreign/sentinel) keeps its valid host pointer
  void genArenaReadbackPtr(const std::vector<char> &buf, char *hostSlot, const uint64_t aOff) {
    const uint64_t devOff = readAt<uint64_t>(buf.data() + aOff), origOff = readAt<uint64_t>(genArenaStaging.data() + aOff);
    if (devOff != origOff)
      if (const auto host = genArenaHostOf(devOff)) std::memcpy(hostSlot, &*host, sizeof(const char *));
  }

  // copy a typed prefix's mutated members from the downloaded arena back to the host, recursing embedded
  // structs; a pointer member holds an arena offset, translated back via genArenaReadbackPtr
  void genArenaReadbackWalk(const std::vector<char> &buf, char *hostBase, const uint64_t arenaOff, const size_t hostOff, const size_t count,
                            const runtime::TypeLayout &l) {
    for (size_t idx = 0; idx < count; ++idx)
      for (size_t m = 0; m < l.memberCount; ++m) {
        const auto member = l.members[m];
        if (member.sizeInBytes == 0 || member.readOnly) continue;
        const size_t hOff = hostOff + idx * l.sizeInBytes + member.offsetInBytes;
        const uint64_t aOff = arenaOff + idx * l.sizeInBytes + member.offsetInBytes;
        if (member.ptrIndirection > 0) genArenaReadbackPtr(buf, hostBase + hOff, aOff);
        else if (member.type && member.type->memberCount > 0) genArenaReadbackWalk(buf, hostBase, aOff, hOff, 1, *member.type);
        else std::memcpy(hostBase + hOff, buf.data() + aOff, member.sizeInBytes);
      }
  }

  // scatter one downloaded rec back to its host, translating pointer slots; `buf` must be indexable at r.off
  void genArenaScatterRec(const std::vector<char> &buf, const GenArenaRec &r) {
    if (r.readOnly) return; // const input: never copy back
    char *host = const_cast<char *>(r.host);
    if (r.indirection > 1) {
      // an array of `count` pointers (each an arena offset); translate the ones the kernel changed
      for (size_t i = 0; i < r.count; ++i)
        genArenaReadbackPtr(buf, host + i * sizeof(uintptr_t), r.off + i * sizeof(uintptr_t));
      return;
    }
    if (r.l->memberCount == 0) std::memcpy(host, buf.data() + r.off, r.count * r.l->sizeInBytes);
    else genArenaReadbackWalk(buf, host, r.off, 0, r.count, *r.l);
    // a downcast widens the object past its static pointee type (a `_List_node_base*` link reaches a
    // `_List_node<int>` tail beyond the prefix); copy the remainder wholesale, it holds no pointer slot
    if (const size_t typed = r.count * r.l->sizeInBytes; r.size > typed)
      std::memcpy(host + typed, buf.data() + r.off + typed, r.size - typed);
  }

  void genArenaReadback() {
    if (!genArenaActive || genArenaRecs.empty()) return;
    if (genArenaPersist) {
      // POLYRT_ARENA_EAGER: diagnostic/conservative - copy this kernel's touched outputs straight back, so
      // correctness does not depend on map_read coverage (isolates upload bugs from the lazy-readback path)
      static const bool eager = std::getenv(env::PolyrtArenaEager) != nullptr;
      if (eager) {
        std::vector<char> buf(genArenaStaging.size());
        remoteRead(buf.data(), genArenaDevBase, 0, buf.size());
        for (const uint64_t off : genArenaTouched)
          for (const auto &r : genArenaRecs)
            if (r.off == off) genArenaScatterRec(buf, r);
        return;
      }
      // lazy: the kernel's outputs stay device-resident and flow to the next kernel on-device; just flag them
      // so a later host read (map_read) pulls the touched leaf home. nothing is copied at the launch boundary
      for (const uint64_t off : genArenaTouched)
        for (auto &r : genArenaRecs)
          if (r.off == off && !r.readOnly) r.deviceDirty = true;
      return;
    }
    std::vector<char> buf(genArenaStaging.size());
    remoteRead(buf.data(), genArenaDevBase, 0, buf.size());
    for (const auto &r : genArenaRecs)
      genArenaScatterRec(buf, r);
  }

  // the host is about to read `p`: pull the device-dirty leaf it falls in back home (the lazy half of persist)
  void genArenaMapRead(const void *p) {
    if (!genArenaPersist) return;
    GenArenaRec *r = genArenaRecOf(p);
    if (!r || !r->deviceDirty) return;
    char *host = const_cast<char *>(r->host);
    if (r->indirection == 1 && r->l->memberCount == 0) remoteRead(host, genArenaDevBase, r->off, r->size); // flat: straight copy
    else { // deep: needs the arena-indexed buf for pointer-slot translation
      std::vector<char> buf(r->off + r->size);
      remoteRead(buf.data() + r->off, genArenaDevBase, r->off, r->size);
      genArenaScatterRec(buf, *r);
    }
    r->deviceDirty = false;
  }

  // the host is about to write `p`: it becomes authoritative, so re-send its leaf before the next kernel reads it
  void genArenaMapWrite(const void *p) {
    if (!genArenaPersist) return;
    if (GenArenaRec *r = genArenaRecOf(p)) {
      r->hostDirty = true;
      r->deviceDirty = false;
    }
  }

  uintptr_t mirrorGraph(const void *root, const size_t *ptrOffsets, const size_t nPtr) {
    if (!root) return 0;
    std::vector<size_t> offs(ptrOffsets, ptrOffsets + nPtr);
    std::sort(offs.begin(), offs.end()); // read-back aligns origPtrs with this order
    MirroredGraph g{offs, {}, {}, {}};
    const auto deviceAddrOf = [&](const char *p) -> uintptr_t {
      if (!p) return 0;
      if (const auto sma = queryLocal(p)) return sma->first->remote.ptr + sma->second;
      const PtrQuery meta = localReflect(p);
      if (meta.sizeInBytes == 0) return 0; // foreign / untracked
      const char *base = p - meta.offsetInBytes;
      const uintptr_t dev = createDeviceAllocation(base, meta.sizeInBytes, nullptr, meta.hostReadOnly);
      remoteWrite(base, dev, 0, meta.sizeInBytes);
      return dev + meta.offsetInBytes;
    };
    std::unordered_set<const char *> recorded;
    std::vector<const char *> work{static_cast<const char *>(root)};
    while (!work.empty()) {
      const char *p = work.back();
      work.pop_back();
      const PtrQuery meta = localReflect(p);
      if (meta.sizeInBytes == 0) continue; // foreign
      const char *base = p - meta.offsetInBytes;
      if (!recorded.insert(base).second) continue;
      g.nodes.push_back(base);
      g.devs.push_back(deviceAddrOf(base));
      for (const size_t off : offs)
        if (const char *nb = readPtrValue(base + off)) work.push_back(nb);
    }
    g.origPtrs.resize(g.nodes.size());
    for (size_t i = 0; i < g.nodes.size(); ++i) {
      g.origPtrs[i].resize(offs.size());
      for (size_t k = 0; k < offs.size(); ++k) {
        const uintptr_t dq = deviceAddrOf(readPtrValue(g.nodes[i] + offs[k]));
        if (dq) mirrorPatch(g.devs[i], offs[k], dq);
        // the slot now holds dq (patched) or the uploaded host bytes (foreign/null pointee, unpatched)
        g.origPtrs[i][k] = dq ? dq : reinterpret_cast<uintptr_t>(readPtrValue(g.nodes[i] + offs[k]));
      }
    }
    if (debug) std::fprintf(stderr, "[SMA] mirrorGraph(root=%p, nPtr=%zu) -> %zu nodes\n", root, nPtr, g.nodes.size());
    const uintptr_t rootDev = deviceAddrOf(static_cast<const char *>(root));
    mirroredGraphs[root] = std::move(g);
    return rootDev;
  }

  void unmirrorReadGraph(const void *root) {
    const auto it = mirroredGraphs.find(root);
    if (it == mirroredGraphs.end()) return;
    const MirroredGraph &g = it->second;
    const std::vector<size_t> &offs = g.ptrOffsets; // sorted at mirror time; aligns with origPtrs
    std::vector<size_t> sizes(g.nodes.size());
    for (size_t i = 0; i < g.nodes.size(); ++i)
      sizes[i] = localReflect(g.nodes[i]).sizeInBytes;
    // inverse of mirrorGraph's deviceAddrOf: a device address back to the host interior it mirrors
    const auto hostOf = [&](const uintptr_t dev) -> std::optional<const char *> {
      if (!dev) return static_cast<const char *>(nullptr);
      for (size_t j = 0; j < g.devs.size(); ++j)
        if (sizes[j] && dev >= g.devs[j] && dev < g.devs[j] + sizes[j]) return g.nodes[j] + (dev - g.devs[j]);
      return std::nullopt;
    };
    for (size_t i = 0; i < g.nodes.size(); ++i) {
      const size_t sz = sizes[i];
      if (!sz) continue;
      std::vector<char> buf(sz);
      remoteRead(buf.data(), g.devs[i], 0, sz);
      char *host = const_cast<char *>(g.nodes[i]);
      size_t pos = 0;
      for (size_t k = 0; k < offs.size(); ++k) {
        const size_t off = offs[k];
        if (off + sizeof(uintptr_t) > sz) break;
        if (off > pos) std::memcpy(host + pos, buf.data() + pos, off - pos); // non-pointer gap
        // a pointer slot holds a device address; translate it back only when the kernel changed it, so a
        // slot the device never wrote (a foreign/sentinel pointer) keeps its valid host pointer
        const uintptr_t devVal = readAt<uintptr_t>(buf.data() + off);
        if (devVal != g.origPtrs[i][k])
          if (const auto h = hostOf(devVal)) {
            const char *hp = *h;
            std::memcpy(host + off, &hp, sizeof(uintptr_t));
          }
        pos = off + sizeof(uintptr_t);
      }
      if (pos < sz) std::memcpy(host + pos, buf.data() + pos, sz - pos);
    }
  }
};

} // namespace polyregion::polyrt
