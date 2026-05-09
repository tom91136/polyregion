#pragma once

#include <string_view>
#include <unordered_map>
#include <vector>

#include "polyinvoke/runtime.h"

namespace polyregion::test_utils {
using ImageGroups = std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>>;

std::unique_ptr<invoke::Platform> makePlatform(invoke::Backend backend);

std::vector<std::pair<std::string, std::string>> findTestImage(const ImageGroups &images, const invoke::Backend &backend,
                                                               const std::vector<std::string> &features);

bool isBackendDisabled(invoke::Backend backend);

bool isDeviceDisabled(std::string_view deviceName);

} // namespace polyregion::test_utils
