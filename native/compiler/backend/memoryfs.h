#pragma once

#include <optional>
#include <string>

namespace polyregion::memoryfs {

std::optional<std::string> open(const std::string &name);
bool close(const std::string &name);

} // namespace polyregion::memoryfs