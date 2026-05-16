#pragma once

#include <string_view>
#include <unordered_map>
#include <vector>

#include "polyinvoke/device_lock.h"
#include "polyinvoke/runtime.h"

namespace polyregion::test_utils {
using ImageGroups = std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>>;

std::unique_ptr<invoke::Platform> makePlatform(invoke::Backend backend);

// Cross-process exclusive lock per device so parallel ctest runners do not race on the same hardware.
invoke::DeviceLock lockDevice(invoke::Backend backend, invoke::Device &device);

std::vector<std::pair<std::string, std::string>> findTestImage(const ImageGroups &images, const invoke::Backend &backend,
                                                               const std::vector<std::string> &features);

bool isBackendDisabled(invoke::Backend backend);

bool isDeviceDisabled(std::string_view deviceName);

} // namespace polyregion::test_utils
