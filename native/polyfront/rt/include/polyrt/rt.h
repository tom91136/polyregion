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

POLYREGION_EXPORT extern std::unique_ptr<Platform> currentPlatform;
POLYREGION_EXPORT extern std::unique_ptr<Device> currentDevice;
POLYREGION_EXPORT extern std::unique_ptr<DeviceQueue> currentQueue;

enum class DebugLevel : uint8_t { None = 0, Info = 1, Debug = 2, Trace = 3 };

POLYREGION_EXPORT void initialise();
POLYREGION_EXPORT bool hostFallback();
POLYREGION_EXPORT DebugLevel debugLevel();
POLYREGION_EXPORT __attribute__((format(printf, 2, 3)))void log(DebugLevel level, const char *fmt, ...);
POLYREGION_EXPORT bool loadKernelObject(const char *moduleName, const KernelObject &object);
POLYREGION_EXPORT void dispatchHostThreaded(size_t global, void *functorData, const char *moduleId);
POLYREGION_EXPORT void dispatchManaged(size_t global, size_t local, size_t localMemBytes, void *functorData, const char *moduleId);

POLYREGION_EXPORT bool loadKernelObject(const char *moduleName, const KernelObject &object);

} // namespace polyregion::polyrt

extern "C" {
POLYREGION_EXPORT void polyrt_map_read(void *origin, ptrdiff_t sizeInBytes, size_t unitInBytes);
POLYREGION_EXPORT void polyrt_map_write(void *origin, ptrdiff_t sizeInBytes, size_t unitInBytes);
POLYREGION_EXPORT void polyrt_map_readwrite(void *origin, ptrdiff_t sizeInBytes, size_t unitInBytes);

POLYREGION_EXPORT void *polyrt_usm_malloc(size_t size);
POLYREGION_EXPORT void *polyrt_usm_calloc(size_t nmemb, size_t size);
POLYREGION_EXPORT void *polyrt_usm_realloc(void *ptr, size_t size);
POLYREGION_EXPORT void *polyrt_usm_memalign(size_t /*alignment*/, size_t size);
POLYREGION_EXPORT void *polyrt_usm_aligned_alloc(size_t /*alignment*/, size_t size);
POLYREGION_EXPORT void polyrt_usm_free(void *ptr);

POLYREGION_EXPORT void *polyrt_usm_operator_new(size_t size);
POLYREGION_EXPORT void polyrt_usm_operator_delete(void *ptr);
POLYREGION_EXPORT void polyrt_usm_operator_delete_sized(void *ptr, size_t size);
}
