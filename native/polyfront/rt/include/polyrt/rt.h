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

POLYREGION_EXPORT void initialise();
POLYREGION_EXPORT bool hostFallback();
POLYREGION_EXPORT bool loadKernelObject(const char *moduleName, const KernelObject &object);
POLYREGION_EXPORT void dispatchHostThreaded(size_t global, void *functorData, const char *moduleId);
POLYREGION_EXPORT void dispatchManaged(size_t global, size_t local, size_t localMemBytes, void *functorData, const char *moduleId);

POLYREGION_EXPORT bool loadKernelObject(const char *moduleName, const KernelObject &object);

} // namespace polyregion::polyrt
