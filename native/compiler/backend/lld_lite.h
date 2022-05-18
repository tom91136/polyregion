#pragma once

#include "InputFiles.h"
#include "lld/Common/DWARF.h"
#include "lld/Common/Driver.h"
#include "llvm/Object/ELF.h"

namespace polyregion::backend::lld_lite {

std::pair<std::optional<std::string>, std::optional<std::string>> link(const std::vector<std::string> &args,
                                                                       const std::vector<lld::elf::InputFile *> &files);

} // namespace polyregion::backend::lld_lite