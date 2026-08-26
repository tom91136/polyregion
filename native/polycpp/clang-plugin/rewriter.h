#pragma once

#include <memory>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"

#include "polyfront/options_backend.hpp"

namespace polyregion::polystl {

[[nodiscard]] std::unique_ptr<clang::ASTConsumer> makeOffloadRewriteConsumer(clang::CompilerInstance &CI,
                                                                             const polyregion::polyfront::Options &opts,
                                                                             std::shared_ptr<std::vector<int8_t>> packageProgramBitcode);

} // namespace polyregion::polystl
