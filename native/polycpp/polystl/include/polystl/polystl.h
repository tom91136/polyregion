#pragma once

#include <cstddef>
#include <cstring>

#include "concurrency_utils.hpp"
#include "polyrt/runtime.h"

namespace polystl {

using namespace polyregion::runtime;

POLYREGION_EXPORT std::unique_ptr<Platform> createPlatform();
POLYREGION_EXPORT std::unique_ptr<Device> selectDevice(Platform &p);

extern std::unique_ptr<Platform> thePlatform;
extern std::unique_ptr<Device> theDevice;
extern std::unique_ptr<DeviceQueue> theQueue;

POLYREGION_EXPORT std::optional<polystl::PlatformKind> platformKind();
POLYREGION_EXPORT bool dispatchHostThreaded(size_t global, void *functorData, const KernelObject &object);
POLYREGION_EXPORT bool dispatchManaged(size_t global, size_t local, size_t localMemBytes, //
                                       size_t functorDataSize, const void *functorData,         //
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

POLYREGION_EXPORT [[nodiscard]] void *__polyregion_malloc(size_t size); // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT void __polyregion_free(void *ptr);                    // NOLINT(*-reserved-identifier)
