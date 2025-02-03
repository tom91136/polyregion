#pragma once

#include "polyfront/options_backend.hpp"

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"

namespace polyregion::polystl {

class OffloadRewriteConsumer : public clang::ASTConsumer {
  clang::CompilerInstance &CI;
  polyregion::polyfront::Options opts;

public:
  OffloadRewriteConsumer(clang::CompilerInstance &CI, const polyregion::polyfront::Options &opts);
  void HandleTranslationUnit(clang::ASTContext &C) override;
};

} // namespace polyregion::polystl