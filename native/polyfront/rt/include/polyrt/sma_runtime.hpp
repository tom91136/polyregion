#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "polyregion/env_keys.h"
#include "polyregion/show.hpp"
#include "polyregion/types.h"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"

namespace polyregion::polyrt::sma {

// Arena: generic single-arena marshalling - the capture graph is laid into one device arena and the kernel
// (arena-lowered) derefs arena[off]; a non-arena-lowered kernel would mis-deref the offsets
enum class MirrorMode { Compiletime, Runtime, Off, Arena };

POLYREGION_RT_PROTECT inline MirrorMode mirrorMode() {
  static const MirrorMode mode = [] {
    const auto *v = std::getenv(env::PolyrtMirror);
    if (!v) return MirrorMode::Compiletime;
    const std::string_view s(v);
    if (s == "runtime") return MirrorMode::Runtime;
    if (s == "off") return MirrorMode::Off;
    if (s == "arena") return MirrorMode::Arena;
    return MirrorMode::Compiletime;
  }();
  return mode;
}

// binding-slot device formats have no flat address space (or use SVM-off), so the arena is their
// default marshaller; physical formats (PTX/HSACO) keep the mirror. an explicit POLYRT_MIRROR still wins
POLYREGION_RT_PROTECT inline bool arenaFormat(const invoke::ModuleFormat fmt) {
  return fmt == invoke::ModuleFormat::Source || fmt == invoke::ModuleFormat::SPIRV_GLCompute || fmt == invoke::ModuleFormat::SPIRV_Kernel;
}
// SPIR-V backends (no int->ptr in llvm.cpp) read the arena through a fixed roster of typed view descriptors
// (ArenaView); flat c_source uses the byte form. keep arenaViewCount in sync with ArenaView.viewTpes (7: i8..f16)
inline constexpr int arenaViewCount = 7;
POLYREGION_RT_PROTECT inline bool arenaViewForm(const invoke::ModuleFormat fmt) {
  return fmt == invoke::ModuleFormat::SPIRV_GLCompute || fmt == invoke::ModuleFormat::SPIRV_Kernel;
}
POLYREGION_RT_PROTECT inline MirrorMode mirrorModeFor(const invoke::ModuleFormat fmt) {
  if (std::getenv(env::PolyrtMirror)) return mirrorMode();
  return arenaFormat(fmt) ? MirrorMode::Arena : MirrorMode::Compiletime;
}

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
  if (debugLevel() < DebugLevel::Debug) return;
  log(DebugLevel::Debug, "[Allocations (%lu)]", sma.localToRemoteAlloc.size());
  for (auto [k, v] : sma.localToRemoteAlloc) {
    const auto &l = sma.remoteToLocalPtr[v.remote.ptr];
    log(DebugLevel::Debug, "\t[Local(0x%jx, %4ld) -> Remote(0x%jx, %4ld) %10s]", //
        k, l.sizeInBytes, v.remote.ptr, v.remote.sizeInBytes, v.layout ? v.layout->name : "???");
  }
}

template <typename SMA> POLYREGION_RT_PROTECT inline void dumpAllocations(SMA &sma) {
  if (debugLevel() < DebugLevel::Debug) return;
  for (auto [k, v] : sma.localToRemoteAlloc) {
    if (!v.layout || v.layout->memberCount == 0) continue;
    dumpMemoryWithLayout(v.layout, reinterpret_cast<const char *>(k));
  }
}

template <typename SMA>
POLYREGION_RT_PROTECT inline void mapRead(SMA &sma, void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  log(DebugLevel::Debug, "polyrt_map_read(%p, %ld, %ld)", origin, sizeInBytes, unitInBytes);
  if (sizeInBytes == 0) return;
  sma.genArenaMapRead(origin);
  sma.syncRemoteToLocal(origin, sizeInBytes);
  dumpAllocations(sma);
}

template <typename SMA>
POLYREGION_RT_PROTECT inline void mapWrite(SMA &sma, void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  log(DebugLevel::Debug, "polyrt_map_write(%p, %ld, %ld)", origin, sizeInBytes, unitInBytes);
  if (sizeInBytes == 0) return;
  sma.genArenaMapWrite(origin);
  sma.invalidateLocal(origin);
}

template <typename SMA>
POLYREGION_RT_PROTECT inline void mapReadwrite(SMA &sma, void *origin, const ptrdiff_t sizeInBytes, const size_t unitInBytes) {
  log(DebugLevel::Debug, "polyrt_map_readwrite(%p, %ld, %ld)", origin, sizeInBytes, unitInBytes);
  if (sizeInBytes == 0) return;
  sma.genArenaMapRead(origin);
  sma.syncRemoteToLocal(origin);
  sma.genArenaMapWrite(origin);
  sma.invalidateLocal(origin);
}

} // namespace polyregion::polyrt::sma

#define POLYREGION_SMA_ABI_SYMBOL_CHECK(constName, symbol) static_assert(sizeof(&symbol) > 0, #symbol);

#define POLYREGION_DEFINE_SMA_MIRROR_ABI(allocations)                                                                                      \
  POLYREGION_EXPORT extern "C" uintptr_t polyrt_sma_alloc(const void *local, const size_t sizeInBytes, const int hostReadOnly) {           \
    return (allocations).mirrorAlloc(local, sizeInBytes, hostReadOnly != 0);                                                               \
  }                                                                                                                                        \
  POLYREGION_EXPORT extern "C" uintptr_t polyrt_sma_ensure(const void *local) { return (allocations).mirrorEnsure(local); }                \
  POLYREGION_EXPORT extern "C" uintptr_t polyrt_sma_ensure_min(const void *local, const uint64_t minSize) {                                \
    return (allocations).mirrorEnsure(local, minSize);                                                                                     \
  }                                                                                                                                        \
  POLYREGION_EXPORT extern "C" uint64_t polyrt_sma_pointee_size(const void *local) { return (allocations).mirrorPointeeSize(local); }      \
  POLYREGION_EXPORT extern "C" uintptr_t polyrt_sma_ensure_deep(const void *local, const size_t depth) {                                   \
    return (allocations).mirrorEnsureDeep(local, depth);                                                                                   \
  }                                                                                                                                        \
  POLYREGION_EXPORT extern "C" void polyrt_sma_patch(const uintptr_t remote, const size_t offsetInBytes, const uintptr_t value) {          \
    (allocations).mirrorPatch(remote, offsetInBytes, value);                                                                               \
  }                                                                                                                                        \
  POLYREGION_EXPORT extern "C" void polyrt_sma_read_alloc(const void *local) { (allocations).unmirrorReadAlloc(local); }                   \
  POLYREGION_EXPORT extern "C" void polyrt_sma_read_deep(const void *local, const size_t depth) {                                          \
    (allocations).unmirrorReadDeep(local, depth);                                                                                          \
  }                                                                                                                                        \
  POLYREGION_EXPORT extern "C" void polyrt_sma_visit_clear(void) { (allocations).unmirrorVisitClear(); }                                   \
  static thread_local std::vector<size_t> poolPtrOffsets;                                                                                  \
  POLYREGION_EXPORT extern "C" void polyrt_sma_pool_reset(void) { poolPtrOffsets.clear(); }                                                \
  POLYREGION_EXPORT extern "C" void polyrt_sma_pool_ptr(const uint64_t offsetInBytes) { poolPtrOffsets.push_back(offsetInBytes); }         \
  POLYREGION_EXPORT extern "C" uintptr_t polyrt_sma_mirror_graph(const void *root) {                                                       \
    return (allocations).mirrorGraph(root, poolPtrOffsets.data(), poolPtrOffsets.size());                                                  \
  }                                                                                                                                        \
  POLYREGION_EXPORT extern "C" void polyrt_sma_read_graph(const void *root) { (allocations).unmirrorReadGraph(root); }                     \
  namespace { /* XXX clears the SMA free-hook before `allocations` is destroyed (declared after it) */                                     \
  struct ReleaseHookRegistrar {                                                                                                            \
    ReleaseHookRegistrar() {                                                                                                               \
      ::polyregion::rt_reflect::_rt_set_release_cb(                                                                                        \
          +[](void *p) { (allocations).disassociateExact(p, ::polyregion::polyrt::currentDevice != nullptr); });                           \
    }                                                                                                                                      \
    ~ReleaseHookRegistrar() { ::polyregion::rt_reflect::_rt_set_release_cb(nullptr); }                                                     \
  } releaseHookRegistrar;                                                                                                                  \
  }                                                                                                                                        \
  POLYREGION_RUNTIME_ABI(POLYREGION_SMA_ABI_SYMBOL_CHECK)                                                                                  \
  static_assert(true, "swallow trailing semicolon")
