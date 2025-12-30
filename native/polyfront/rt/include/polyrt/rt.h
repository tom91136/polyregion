#pragma once

#include <cstddef>

#include "polyinvoke/runtime.h"

namespace polyregion::polyrt {

using runtime::KernelBundle;
using runtime::KernelObject;
using runtime::TypeLayout;

using invoke::Access;
using invoke::Device;
using invoke::DeviceQueue;
using invoke::Platform;

using invoke::ModuleFormat;
using invoke::PlatformKind;

__RT_PROTECT POLYREGION_EXPORT extern std::unique_ptr<Platform> currentPlatform;
__RT_PROTECT POLYREGION_EXPORT extern std::unique_ptr<Device> currentDevice;
__RT_PROTECT POLYREGION_EXPORT extern std::unique_ptr<DeviceQueue> currentQueue;

enum class DebugLevel : uint8_t { None = 0, Info = 1, Debug = 2, Trace = 3 };

__RT_PROTECT POLYREGION_EXPORT void initialise();
__RT_PROTECT POLYREGION_EXPORT bool hostFallback();
__RT_PROTECT POLYREGION_EXPORT DebugLevel debugLevel();
__RT_PROTECT POLYREGION_EXPORT __attribute__((format(printf, 2, 3))) void log(DebugLevel level, const char *fmt, ...);
__RT_PROTECT POLYREGION_EXPORT bool loadKernelObject(const char *moduleName, const KernelObject &object);
} // namespace polyregion::polyrt

extern "C" {
__RT_PROTECT POLYREGION_EXPORT void polyrt_map_read(void *origin, ptrdiff_t sizeInBytes, size_t unitInBytes);
__RT_PROTECT POLYREGION_EXPORT void polyrt_map_write(void *origin, ptrdiff_t sizeInBytes, size_t unitInBytes);
__RT_PROTECT POLYREGION_EXPORT void polyrt_map_readwrite(void *origin, ptrdiff_t sizeInBytes, size_t unitInBytes);

__RT_PROTECT POLYREGION_EXPORT void *polyrt_usm_malloc(size_t size);
__RT_PROTECT POLYREGION_EXPORT void *polyrt_usm_calloc(size_t nmemb, size_t size);
__RT_PROTECT POLYREGION_EXPORT void *polyrt_usm_realloc(void *ptr, size_t size);
__RT_PROTECT POLYREGION_EXPORT void *polyrt_usm_memalign(size_t /*alignment*/, size_t size);
__RT_PROTECT POLYREGION_EXPORT void *polyrt_usm_aligned_alloc(size_t /*alignment*/, size_t size);
__RT_PROTECT POLYREGION_EXPORT void polyrt_usm_free(void *ptr);

__RT_PROTECT POLYREGION_EXPORT void *polyrt_usm_operator_new(size_t size);
__RT_PROTECT POLYREGION_EXPORT void polyrt_usm_operator_delete(void *ptr);
__RT_PROTECT POLYREGION_EXPORT void polyrt_usm_operator_delete_sized(void *ptr, size_t size);
}
