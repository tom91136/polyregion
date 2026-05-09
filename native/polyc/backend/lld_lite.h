#pragma once

#include <optional>
#include <string>
#include <vector>

#include "llvm/Support/MemoryBufferRef.h"

namespace polyregion::backend::lld_lite {

std::pair<std::optional<std::string>, std::optional<std::string>> linkElf(const std::vector<std::string> &args,
                                                                          const std::vector<llvm::MemoryBufferRef> &files);

} // namespace polyregion::backend::lld_lite