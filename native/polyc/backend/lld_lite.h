#pragma once

#include "llvm/Support/MemoryBufferRef.h"
#include <optional>
#include <string>
#include <vector>

namespace polyregion::backend::lld_lite {

std::pair<std::optional<std::string>, std::optional<std::string>>
linkElf(const std::vector<std::string> &args, const std::vector<llvm::MemoryBufferRef> &files);

} // namespace polyregion::backend::lld_lite