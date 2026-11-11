#pragma once

#include <string_view>

namespace polyregion {

constexpr std::string_view hostOs() {
#ifdef _WIN32
  return "windows";
#elif defined(__APPLE__)
  return "darwin";
#else
  return "linux";
#endif
}

} // namespace polyregion
