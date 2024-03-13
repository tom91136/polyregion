#pragma once

#include <cstddef>
#include <cstring>

#include "concurrency_utils.hpp"
#include "polyrt/runtime.h"

using namespace polyregion::runtime;

POLYREGION_EXPORT extern std::unique_ptr<Platform> __polyregion_selected_platform; // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT extern std::unique_ptr<Device> __polyregion_selected_device;     // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT extern std::unique_ptr<DeviceQueue> __polyregion_selected_queue; // NOLINT(*-reserved-identifier)

POLYREGION_EXPORT extern "C" void __polyregion_initialise_runtime();              // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT extern "C" bool __polyregion_platform_kind(PlatformKind &kind); // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT extern "C" bool __polyregion_dispatch_hostthreaded(             // NOLINT(*-reserved-identifier)
    size_t global, void *functorData, const KernelObject &object);
POLYREGION_EXPORT extern "C" bool __polyregion_dispatch_managed( // NOLINT(*-reserved-identifier)
    size_t global, size_t local, size_t localMemBytes, size_t functorDataSize,
    const void *functorData, //
    const KernelObject &object);

[[nodiscard]] uint64_t __polyregion_builtin_gpu_global_idx(uint32_t); // NOLINT(*-reserved-identifier)

extern "C" inline  KernelBundle __polyregion_deserialise(// NOLINT(*-reserved-identifier)
    size_t size, const unsigned char *data) {
  return KernelBundle::fromMsgPack(size, data);
}

template <polyregion::runtime::PlatformKind Kind, typename F>
const polyregion::runtime::KernelBundle &__polyregion_offload__([[maybe_unused]] F) { // NOLINT(*-reserved-identifier)
  [[maybe_unused]] size_t __stub_kernelImageSize__{};                                 // NOLINT(*-reserved-identifier)
  [[maybe_unused]] const unsigned char *__stub_kernelImageBytes__{};                  // NOLINT(*-reserved-identifier)
  const static KernelBundle bundle = __polyregion_deserialise(__stub_kernelImageSize__, __stub_kernelImageBytes__);
  return bundle;
}



extern "C" inline __attribute__((used)) void *__polyregion_malloc(size_t size) { // NOLINT(*-reserved-identifier)
  __polyregion_initialise_runtime();
  if (!__polyregion_selected_platform || !__polyregion_selected_device || !__polyregion_selected_queue) {
    fprintf(stderr, "[POLYSTL] No device/queue in %s\n", __func__);
    return nullptr;
  }
  if (auto ptr = __polyregion_selected_device->mallocShared(size, polyregion::runtime::Access::RW); ptr) {
    return *ptr;
  } else {
    fprintf(stderr, "[POLYSTL] No USM support in %s\n", __func__);
    return nullptr;
  }
}

extern "C" inline __attribute__((used)) void __polyregion_free(void *ptr) { // NOLINT(*-reserved-identifier)
  __polyregion_initialise_runtime();
  if (!__polyregion_selected_platform || !__polyregion_selected_device || !__polyregion_selected_queue) {
    fprintf(stderr, "[POLYSTL] No device/queue in %s\n", __func__);
  }
  __polyregion_selected_device->freeShared(ptr);
}

extern "C" inline __attribute__((used)) void *__polyregion_operator_new(size_t size) { // NOLINT(*-reserved-identifier)
  return __polyregion_malloc(size);
}
extern "C" inline __attribute__((used)) void __polyregion_operator_delete(void *ptr) { // NOLINT(*-reserved-identifier)
  __polyregion_free(ptr);
}