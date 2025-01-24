#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <unordered_map>

#include "polyregion/types.h"

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
class SynchronisedAllocation {

  struct RangedAllocation {
    uintptr_t ptr;
    size_t sizeInBytes;
  };

  struct RemoteAllocation {
    RangedAllocation remote;
    const runtime::TypeLayout *layout;
    bool localModified;
  };

  std::unordered_map<uintptr_t, RemoteAllocation> localToRemoteAlloc{};
  std::unordered_map<uintptr_t, RangedAllocation> remoteToLocalPtr{};

  LocalReflect localReflect;
  RemoteAlloc remoteAlloc;
  RemoteRead remoteRead;
  RemoteWrite remoteWrite;
  RemoteRelease remoteRelease;

  static char *readPtrValue(const char *base) {
    char *ptr{};
    std::memcpy(&ptr, base, sizeof(uintptr_t));
    return ptr;
  }

  template <typename V, typename F>
  static std::optional<std::pair<V *, size_t>> offsetQuery(std::unordered_map<uintptr_t, V> &map, const void *p, F selectSize) {
    const auto localPtr = reinterpret_cast<uintptr_t>(p);
    if (const auto it = map.find(localPtr); it != map.end()) return std::pair{&it->second, 0};
    for (auto it = map.begin(); it != map.end(); ++it) {
      const uintptr_t base = it->first;
      if (const size_t size = selectSize(it->second); localPtr >= base && localPtr < base + size) {
        return std::pair{&it->second, localPtr - base};
      }
    }
    return {};
  }

  uintptr_t createDeviceAllocation(const char *local, size_t sizeInBytes, const runtime::TypeLayout *s) {
    const uintptr_t remotePtr = remoteAlloc(sizeInBytes);
    localToRemoteAlloc.emplace(reinterpret_cast<uintptr_t>(local),
                               RemoteAllocation{.remote = RangedAllocation{.ptr = remotePtr,            //
                                                                           .sizeInBytes = sizeInBytes}, //
                                                .layout = s,                                            //
                                                .localModified = false});
    remoteToLocalPtr.emplace(remotePtr, RangedAllocation{.ptr = reinterpret_cast<uintptr_t>(local), .sizeInBytes = sizeInBytes});
    return remotePtr;
  }

  void writeSubObject(const char *hostData, const size_t memberOffsetInBytes, const size_t indirection, const uintptr_t devicePtr,
                      const runtime::TypeLayout &tl, const auto metaToCount) {
    fprintf(stderr, "## writeSubObject(hostData=%p, memberOffsetInBytes=%ld, indirection=%ld, devicePtr=%jx, t=@%s)\n", //
            (void *)hostData, memberOffsetInBytes, indirection, devicePtr, tl.name);
    if (char *memberLocalPtr = readPtrValue(hostData + memberOffsetInBytes); !memberLocalPtr) {
      constexpr uintptr_t nullValue = 0;
      remoteWrite(&nullValue, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else if (const PtrQuery meta = localReflect(memberLocalPtr); meta.sizeInBytes != 0) {
      fprintf(stderr, "## localReflect(%p) = {size=%ld, offset=%ld}\n", static_cast<void *>(memberLocalPtr), meta.sizeInBytes,
              meta.offsetInBytes); //

      const uintptr_t memberRemotePtr = mirrorToRemote(memberLocalPtr, indirection, metaToCount(meta), tl);
      remoteWrite(&memberRemotePtr, devicePtr, memberOffsetInBytes, sizeof(uintptr_t));
    } else {
      std::fprintf(stderr, "Foreign pointer %p\n", static_cast<void *>(memberLocalPtr));
    }
  }

  uintptr_t writeIndirect(const char *hostData, const uintptr_t devicePtr, const size_t ptrIndirections, const size_t count,
                          const runtime::TypeLayout &tl) {
    fprintf(stderr, "## writeIndirect(hostData=%p, count=%ld,devicePtr=%jx, t=@ %s, ptrIndirections=%ld)\n", //
            (void *)hostData, count, devicePtr, tl.name, ptrIndirections);
    for (size_t idx = 0; idx < count; ++idx) {
      writeSubObject(hostData, idx * sizeof(uintptr_t), ptrIndirections - 1, devicePtr, tl, [&](auto &meta) {
        const size_t componentSize = ptrIndirections - 1 == 1 ? tl.sizeInBytes : sizeof(uintptr_t);
        return (meta.sizeInBytes - meta.offsetInBytes) / componentSize;
      });
    }
    return devicePtr;
  }

  uintptr_t writeStruct(const char *hostData, const size_t offsetInBytes, const uintptr_t devicePtr, const size_t count,
                        const runtime::TypeLayout &tl, const bool writeBody) {
    fprintf(stderr, "## writeStruct(hostData=%p, offsetInBytes=%zu, count=%ld,devicePtr=%jx, t=@%s, writeBody=%d)\n", //
            (void *)hostData, offsetInBytes, count, devicePtr, tl.name, writeBody);
    if (writeBody) remoteWrite(hostData + offsetInBytes, devicePtr, offsetInBytes, tl.sizeInBytes * count);
    for (size_t idx = 0; idx < count; ++idx) {
      for (size_t m = 0; m < tl.memberCount; ++m) {
        const auto member = tl.members[m];
        const auto memberOffsetInBytes = offsetInBytes + (idx * tl.sizeInBytes + member.offsetInBytes);
        if (member.sizeInBytes == 0) continue;
        if (member.ptrIndirection > 0)
          writeSubObject(hostData, memberOffsetInBytes, member.ptrIndirection, devicePtr, *member.type, [&](auto &meta) {
            const size_t componentSize = member.ptrIndirection == 1 ? member.componentSize : member.sizeInBytes;
            return (meta.sizeInBytes - meta.offsetInBytes) / componentSize;
          });
        else if (member.type->memberCount > 0) {
          writeStruct(hostData, memberOffsetInBytes, devicePtr, 1, *member.type, false);
        }
      }
    }
    return devicePtr;
  }

  uintptr_t mirrorToRemote(const char *hostData, const size_t ptrIndirections, const size_t count, const runtime::TypeLayout &l) {
    fprintf(stderr, "## mirrorToRemote(hostData=%p, ptrIndirections=%zu, count=%zu, t=@%s)\n", //
            (void *)hostData, ptrIndirections, count, l.name);
    if (const auto query = offsetQuery(localToRemoteAlloc, hostData, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      if (!alloc->localModified) return alloc->remote.ptr + offsetInBytes;
      else
        return ptrIndirections > 1 ? writeIndirect(hostData, alloc->remote.ptr, ptrIndirections, count, *alloc->layout)
                                   : writeStruct(hostData, 0, alloc->remote.ptr, count, *alloc->layout, true);
    } else
      return ptrIndirections > 1
                 ? writeIndirect(hostData, createDeviceAllocation(hostData, sizeof(uintptr_t) * count, nullptr), ptrIndirections, count, l)
                 : writeStruct(hostData, 0, createDeviceAllocation(hostData, l.sizeInBytes * count, &l), count, l, true);
  }

public:
  SynchronisedAllocation(LocalReflect localReflect, //
                         RemoteAlloc remoteAlloc,   //
                         RemoteRead remoteRead,     //
                         RemoteWrite remoteWrite,   //
                         RemoteRelease remoteRelease)
      : localReflect(localReflect), //
        remoteAlloc(remoteAlloc),   //
        remoteRead(remoteRead),     //
        remoteWrite(remoteWrite),   //
        remoteRelease(remoteRelease) {}

  [[nodiscard]] uintptr_t syncLocalToRemote(const void *p, const runtime::TypeLayout &s) {
    return mirrorToRemote(static_cast<const char *>(p), 1, 1, s);
  }

  std::optional<uintptr_t> syncLocalToRemote(const void *p) {
    if (const auto query = offsetQuery(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      if (!alloc->localModified) return alloc->remote.ptr + offsetInBytes;
      else
        return writeStruct(static_cast<const char *>(p), 0, alloc->remote.ptr, alloc->remote.sizeInBytes / alloc->layout->sizeInBytes,
                           *alloc->layout, true);
    }
    return {};
  }

  std::optional<uintptr_t> invalidateLocal(const void *p) {
    if (const auto query = offsetQuery(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      alloc->localModified = true;
      return alloc->remote.ptr + offsetInBytes;
    }
    return {};
  }

  void readSubObject(char *p, const size_t memberOffsetInBytes) {
    if (char *memberRemotePtr = readPtrValue(p + memberOffsetInBytes); !memberRemotePtr) {
      std::memset(p + memberOffsetInBytes, 0, sizeof(uintptr_t));
    } else if (const auto query = offsetQuery(remoteToLocalPtr, memberRemotePtr, [](auto &x) { return x.sizeInBytes; })) {
      auto [alloc, _] = *query;
      syncRemoteToLocal(reinterpret_cast<void *>(alloc->ptr));
      std::memcpy(p + memberOffsetInBytes, &alloc->ptr, sizeof(uintptr_t));
    } else {
      std::fprintf(stderr, "Remote introduced foreign member pointer %p\n", static_cast<void *>(memberRemotePtr));
    }
  }

  void readStruct(char *p, const size_t offset, const runtime::TypeLayout *tl) {
    for (size_t m = 0; m < tl->memberCount; ++m) {
      const auto member = tl->members[m];
      if (member.sizeInBytes == 0) continue;
      if (member.ptrIndirection > 0) readSubObject(p, offset + member.offsetInBytes);
      else if (member.type->memberCount > 0) readStruct(p, offset + member.offsetInBytes, member.type);
    }
  }

  std::optional<uintptr_t> syncRemoteToLocal(void *p) {

    if (const auto query = offsetQuery(localToRemoteAlloc, p, [](auto &x) { return x.remote.sizeInBytes; })) {
      const auto [alloc, offsetInBytes] = *query;
      const size_t sizeInBytes = alloc->remote.sizeInBytes - offsetInBytes;
      remoteRead(p, alloc->remote.ptr, offsetInBytes, sizeInBytes);
      if (const runtime::TypeLayout *tl = alloc->layout) {
        for (size_t idx = 0; idx < sizeInBytes / alloc->remote.sizeInBytes; ++idx)
          readStruct(static_cast<char *>(p), idx * alloc->remote.sizeInBytes, tl);
      } else { // array of pointers
        for (size_t idx = 0; idx < sizeInBytes / sizeof(uintptr_t); ++idx)
          readSubObject(static_cast<char *>(p), idx * sizeof(uintptr_t));
      }
      return alloc->remote.ptr + offsetInBytes;
    }
    return {};
  }

  void disassociate(const void *p) {
    auto [it, _] = offsetQuery(localToRemoteAlloc, p);
    if (it != localToRemoteAlloc.end()) {
      remoteRelease(it->second.remotePtr.value);
      remoteToLocalPtr.erase(it->first);
      localToRemoteAlloc.erase(it);
    }
  }
};

} // namespace polyregion::polyrt