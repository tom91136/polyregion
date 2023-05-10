#pragma once

#include "runtime.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace polyregion::test_utils {
using ImageGroups = std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>>;

std::vector<std::pair<std::string, std::string>> findTestImage(const ImageGroups &images,
                                                               const polyregion::runtime::Backend &backend,
                                                               const std::vector<std::string> &features);

} // namespace polyregion::test_utils
