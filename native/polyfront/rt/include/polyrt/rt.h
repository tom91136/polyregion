#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

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

POLYREGION_RT_PROTECT POLYREGION_EXPORT extern std::unique_ptr<Platform> currentPlatform;
POLYREGION_RT_PROTECT POLYREGION_EXPORT extern std::unique_ptr<Device> currentDevice;
POLYREGION_RT_PROTECT POLYREGION_EXPORT extern std::unique_ptr<DeviceQueue> currentQueue;

enum class DebugLevel : uint8_t { None = 0, Info = 1, Debug = 2, Trace = 3 };

#if defined(__GNUC__) || defined(__clang__)
  #define POLYREGION_PRINTF_FORMAT(fmt_index, first_arg) __attribute__((format(printf, fmt_index, first_arg)))
#else
  #define POLYREGION_PRINTF_FORMAT(fmt_index, first_arg)
#endif

POLYREGION_RT_PROTECT POLYREGION_EXPORT void initialise();
POLYREGION_RT_PROTECT POLYREGION_EXPORT bool hostFallback();

POLYREGION_RT_PROTECT POLYREGION_EXPORT void ensureRoSegmentsRecorded();

// XXX Exits with code 77 (autotools "skipped") so test runners can distinguish "no compatible
// target on this device" from a real wrong-output / crash failure.
[[noreturn]] POLYREGION_RT_PROTECT POLYREGION_EXPORT void noCompatibleKernelExit(const char *site);
[[noreturn]] POLYREGION_RT_PROTECT POLYREGION_EXPORT void skipExit(const char *reason);
POLYREGION_RT_PROTECT POLYREGION_EXPORT DebugLevel debugLevel();
POLYREGION_RT_PROTECT POLYREGION_EXPORT POLYREGION_PRINTF_FORMAT(2, 3) void log(DebugLevel level, const char *fmt, ...);
POLYREGION_RT_PROTECT POLYREGION_EXPORT bool loadKernelObject(const char *moduleName, const KernelObject &object,
                                                              const void *capture = nullptr, const TypeLayout *interfaceLayout = nullptr,
                                                              std::string *loadedModuleName = nullptr);
} // namespace polyregion::polyrt

#undef POLYREGION_PRINTF_FORMAT

extern "C" {
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_map_read(void *origin, ptrdiff_t sizeInBytes, size_t unitInBytes);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_map_write(void *origin, ptrdiff_t sizeInBytes, size_t unitInBytes);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_map_readwrite(void *origin, ptrdiff_t sizeInBytes, size_t unitInBytes);

POLYREGION_RT_PROTECT POLYREGION_EXPORT void *polyrt_usm_malloc(size_t size);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void *polyrt_usm_aligned_alloc(size_t /*alignment*/, size_t size);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_usm_free(void *ptr);

POLYREGION_RT_PROTECT POLYREGION_EXPORT void *polyrt_usm_operator_new(size_t size);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_usm_operator_delete(void *ptr);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_usm_operator_delete_sized(void *ptr, size_t size);

POLYREGION_RT_PROTECT POLYREGION_EXPORT void *polyrt_record_malloc(size_t size);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_record_free(void *ptr);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void *polyrt_record_aligned_alloc(size_t alignment, size_t size);

POLYREGION_RT_PROTECT POLYREGION_EXPORT void *polyrt_record_operator_new(size_t size);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_record_operator_delete(void *ptr);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_record_operator_delete_sized(void *ptr, size_t size);

POLYREGION_RT_PROTECT POLYREGION_EXPORT uintptr_t polyrt_sma_alloc(const void *local, size_t sizeInBytes, int hostReadOnly);
POLYREGION_RT_PROTECT POLYREGION_EXPORT uintptr_t polyrt_sma_ensure(const void *local);
POLYREGION_RT_PROTECT POLYREGION_EXPORT uintptr_t polyrt_sma_ensure_min(const void *local, uint64_t minSize);
POLYREGION_RT_PROTECT POLYREGION_EXPORT uintptr_t polyrt_sma_ensure_deep(const void *local, size_t depth);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_sma_patch(uintptr_t remote, size_t offsetInBytes, uintptr_t value);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_sma_read_alloc(const void *local);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_sma_read_deep(const void *local, size_t depth);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_sma_visit_clear(void);

POLYREGION_RT_PROTECT POLYREGION_EXPORT uintptr_t polyrt_sma_mirror_graph(const void *root);
POLYREGION_RT_PROTECT POLYREGION_EXPORT void polyrt_sma_read_graph(const void *root);
}
