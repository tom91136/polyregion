#pragma once

#include <string>
#include <string_view>

#include "llvm/Support/xxhash.h"

#include "fmt/format.h"

namespace polyregion::mirror {

inline std::string idFor(std::string_view moduleName) {
  return fmt::format("{:016x}", llvm::xxh3_64bits(llvm::StringRef(moduleName.data(), moduleName.size())));
}

inline std::string preludeName(std::string_view id) { return "__polyregion_mirror_prelude_" + std::string(id); }
inline std::string postludeName(std::string_view id) { return "__polyregion_mirror_postlude_" + std::string(id); }

} // namespace polyregion::mirror
