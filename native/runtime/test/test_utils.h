#pragma once

#include "runtime.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace polyregion::test_utils {

std::optional<std::string>
findTestImage(const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>> &images,
              const polyregion::runtime::Backend &backend, const std::vector<std::string> &features);

} // namespace polyregion::test_utils
