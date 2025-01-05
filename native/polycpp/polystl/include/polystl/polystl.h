#pragma once

#include "polyrt/runtime.h"

#include <cassert>
#include <cstddef>
#include <cstring>

using polyregion::runtime::Access;
using polyregion::runtime::Device;
using polyregion::runtime::DeviceQueue;
using polyregion::runtime::KernelBundle;
using polyregion::runtime::KernelObject;
using polyregion::runtime::ModuleFormat;
using polyregion::runtime::Platform;
using polyregion::runtime::PlatformKind;
using polyregion::runtime::TypeLayout;

static constexpr const char *__polyregion_prefix = "PolySTL"; // NOLINT(*-reserved-identifier)

#ifdef POLYSTL_LOG
  #error Trace already defined
#else

  #define POLYSTL_LOG(fmt, ...) std::fprintf(stderr, "[%s] " fmt "\n", __polyregion_prefix, __VA_ARGS__)
//  #define POLYSTL_LOG(fmt, ...)

#endif

namespace polyregion::polystl {

POLYREGION_EXPORT extern std::unique_ptr<Platform> currentPlatform;
POLYREGION_EXPORT extern std::unique_ptr<Device> currentDevice;
POLYREGION_EXPORT extern std::unique_ptr<DeviceQueue> currentQueue;

POLYREGION_EXPORT void initialise();
POLYREGION_EXPORT PlatformKind platformKind();
POLYREGION_EXPORT bool loadKernelObject(const char *moduleName, const KernelObject &object);
POLYREGION_EXPORT void dispatchHostThreaded(size_t global, void *functorData, const char *moduleId);
POLYREGION_EXPORT void dispatchManaged(size_t global, size_t local, //
                                       size_t localMemBytes,        //
                                       const TypeLayout *layout,
                                       void *functorData, //
                                       const char *moduleId);

POLYREGION_EXPORT bool loadKernelObject(const char *moduleName, const KernelObject &object);
} // namespace polyregion::polystl

template <PlatformKind Kind, typename F> const KernelBundle &__polyregion_offload__([[maybe_unused]] F) { // NOLINT(*-reserved-identifier)
  assert(false && "impl not replaced");
  std::abort();
}

// =========

[[nodiscard]] uint32_t __polyregion_builtin_gpu_global_idx(uint32_t);  // NOLINT(*-reserved-identifier)
[[nodiscard]] uint32_t __polyregion_builtin_gpu_global_size(uint32_t); // NOLINT(*-reserved-identifier)

[[nodiscard]] uint32_t __polyregion_builtin_gpu_group_idx(uint32_t);  // NOLINT(*-reserved-identifier)
[[nodiscard]] uint32_t __polyregion_builtin_gpu_group_size(uint32_t); // NOLINT(*-reserved-identifier)

[[nodiscard]] uint32_t __polyregion_builtin_gpu_local_idx(uint32_t);  // NOLINT(*-reserved-identifier)
[[nodiscard]] uint32_t __polyregion_builtin_gpu_local_size(uint32_t); // NOLINT(*-reserved-identifier)

void __polyregion_builtin_gpu_barrier_global(); // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_barrier_local();  // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_barrier_all();    // NOLINT(*-reserved-identifier)

void __polyregion_builtin_gpu_fence_global(); // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_fence_local();  // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_fence_all();    // NOLINT(*-reserved-identifier)

extern "C" inline __attribute__((used)) void *__polyregion_malloc(size_t size) { // NOLINT(*-reserved-identifier)
  polyregion::polystl::initialise();
  if (!polyregion::polystl::currentPlatform || !polyregion::polystl::currentDevice || !polyregion::polystl::currentQueue) {
    POLYSTL_LOG("No device/queue in %s", __func__);
    return nullptr;
  }
  if (const auto ptr = polyregion::polystl::currentDevice->mallocShared(size, Access::RW); ptr) {
    return *ptr;
  } else {
    POLYSTL_LOG("No USM support in %s", __func__);
    return nullptr;
  }
}

extern "C" inline __attribute__((used)) void __polyregion_free(void *ptr) { // NOLINT(*-reserved-identifier)
  polyregion::polystl::initialise();
  if (!polyregion::polystl::currentPlatform || !polyregion::polystl::currentDevice || !polyregion::polystl::currentQueue) {
    POLYSTL_LOG("No device/queue in %s", __func__);
  }
  polyregion::polystl::currentDevice->freeShared(ptr);
}
extern "C" inline __attribute__((used)) void *__polyregion_operator_new(size_t size) { // NOLINT(*-reserved-identifier)
  return __polyregion_malloc(size);
}
extern "C" inline __attribute__((used)) void __polyregion_operator_delete(void *ptr) { // NOLINT(*-reserved-identifier)
  __polyregion_free(ptr);
}
extern "C" inline __attribute__((used)) void __polyregion_operator_delete_sized(void *ptr, size_t size) { // NOLINT(*-reserved-identifier)
  __polyregion_free(ptr);
}
extern "C" inline __attribute__((used)) void *__polyregion_aligned_alloc(size_t alignment, size_t size) { // NOLINT(*-reserved-identifier)
  // TODO actually align it
  return __polyregion_malloc(size);
}