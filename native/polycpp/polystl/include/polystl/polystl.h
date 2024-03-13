#pragma once

#include <cstddef>
#include <cstring>

#include "concurrency_utils.hpp"
#include "polyrt/runtime.h"

namespace polystl {

using namespace polyregion::runtime;

POLYREGION_EXPORT extern std::unique_ptr<Platform> thePlatform;
POLYREGION_EXPORT extern std::unique_ptr<Device> theDevice;
POLYREGION_EXPORT extern std::unique_ptr<DeviceQueue> theQueue;

POLYREGION_EXPORT void initialiseRuntime();
POLYREGION_EXPORT std::optional<polystl::PlatformKind> platformKind();
POLYREGION_EXPORT bool dispatchHostThreaded(size_t global, void *functorData, const KernelObject &object);
POLYREGION_EXPORT bool dispatchManaged(size_t global, size_t local, size_t localMemBytes, //
                                       size_t functorDataSize, const void *functorData,   //
                                       const KernelObject &object);

} // namespace polystl

[[nodiscard]] uint64_t __polyregion_builtin_gpu_global_idx(uint32_t); // NOLINT(*-reserved-identifier)

template <polyregion::runtime::PlatformKind Kind, typename F>                         //
const polyregion::runtime::KernelBundle &__polyregion_offload__([[maybe_unused]] F) { // NOLINT(*-reserved-identifier)
  [[maybe_unused]] size_t __stub_kernelImageSize__{};                                 // NOLINT(*-reserved-identifier)
  [[maybe_unused]] const unsigned char *__stub_kernelImageBytes__{};                  // NOLINT(*-reserved-identifier)
  const static polystl::KernelBundle bundle = polystl::KernelBundle::fromMsgPack(__stub_kernelImageSize__, __stub_kernelImageBytes__);
  fprintf(stderr, "Load %s ", to_string(bundle.objects[0].kind).data());
  fprintf(stderr, "Load %s ", to_string(bundle.objects[0].format).data());
  return bundle;
}

extern "C" inline __attribute__((used)) void *__polyregion_malloc(size_t size) { // NOLINT(*-reserved-identifier)
  using namespace polystl;
  initialiseRuntime();
  if (!thePlatform || !theDevice || !theQueue) {
    fprintf(stderr, "[POLYSTL] No device/queue in %s\n", __func__);
    return nullptr;
  }
  if (auto ptr = theDevice->mallocShared(size, polyregion::runtime::Access::RW); ptr) {
    return *ptr;
  } else {
    fprintf(stderr, "[POLYSTL] No USM support in %s\n", __func__);
    return nullptr;
  }
}

extern "C" inline __attribute__((used)) void __polyregion_free(void *ptr) { // NOLINT(*-reserved-identifier)
  using namespace polystl;
  initialiseRuntime();
  if (!thePlatform || !theDevice || !theQueue) {
    fprintf(stderr, "[POLYSTL] No device/queue in %s\n", __func__);
  }
  theDevice->freeShared(ptr);
}

extern "C" inline __attribute__((used)) void *__polyregion_operator_new(size_t size) { // NOLINT(*-reserved-identifier)
  return __polyregion_malloc(size);
}
extern "C" inline __attribute__((used)) void __polyregion_operator_delete(void *ptr) { // NOLINT(*-reserved-identifier)
  __polyregion_free(ptr);
}