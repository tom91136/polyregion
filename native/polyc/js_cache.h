#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "polyregion/aliases.h"

namespace polyregion::polypass {

std::optional<Vector<uint8_t>> readJsCache(std::string_view engineTag, std::string_view source);
void writeJsCache(std::string_view engineTag, std::string_view source, const uint8_t *data, size_t size);
String hostArchTag();

} // namespace polyregion::polypass
