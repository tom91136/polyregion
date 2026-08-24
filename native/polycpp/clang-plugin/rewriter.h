#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"

#include "polyfront/options_backend.hpp"

namespace polyregion::polystl {

using ResolvedSymBitcode = std::vector<std::vector<int8_t>>;

class OffloadRewriteConsumer : public clang::ASTConsumer {
  clang::CompilerInstance &CI;
  polyregion::polyfront::Options opts;
  std::shared_ptr<ResolvedSymBitcode> resolvedSymBitcode;

public:
  OffloadRewriteConsumer(clang::CompilerInstance &CI, const polyregion::polyfront::Options &opts,
                         std::shared_ptr<ResolvedSymBitcode> resolvedSymBitcode);
  void HandleTranslationUnit(clang::ASTContext &C) override;
};

} // namespace polyregion::polystl
