#pragma once

#include "polyrt/runtime.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace polyregion::test_utils {
using ImageGroups = std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>>;

std::unique_ptr<runtime::Platform> makePlatform(runtime::Backend backend);

std::vector<std::pair<std::string, std::string>> findTestImage(const ImageGroups &images,
                                                               const runtime::Backend &backend,
                                                               const std::vector<std::string> &features);

} // namespace polyregion::test_utils
