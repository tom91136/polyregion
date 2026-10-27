#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "magic_enum/magic_enum.hpp"
#include "nlohmann/json.hpp"

#include "polyregion/env_keys.h"
#include "polyregion/types.h"

namespace polyregion::polyfront {

inline nlohmann::json imageJson(const runtime::PlatformKind kind, const runtime::ModuleFormat format,
                                const std::vector<std::string> &features) {
  return {{"kind", magic_enum::enum_name(kind)}, {"format", magic_enum::enum_name(format)}, {"features", features}};
}

inline void recordImages(const std::string &module, const nlohmann::json &images) {
  const auto path = std::getenv(env::PolyfrontImageManifest);
  if (!path || !*path) return;
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::CD_OpenAlways, llvm::sys::fs::FA_Write,
                           llvm::sys::fs::OF_Append | llvm::sys::fs::OF_Text);
  if (!ec) out << nlohmann::json{{"module", module}, {"images", images}}.dump() << '\n';
}

inline void recordRemoteImage(const std::string &module, const int32_t kind, const int32_t format,
                              const std::vector<std::string> &features) {
  recordImages(module, nlohmann::json::array(
                           {imageJson(static_cast<runtime::PlatformKind>(kind), static_cast<runtime::ModuleFormat>(format), features)}));
}

} // namespace polyregion::polyfront
