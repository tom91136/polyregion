#pragma once

#include "plugin.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"

namespace polyregion::polystl {

class OffloadRewriteConsumer : public clang::ASTConsumer {
  clang::CompilerInstance &CI;
  Options opts;

public:
  OffloadRewriteConsumer(clang::CompilerInstance &CI, const Options &opts);
  void HandleTranslationUnit(clang::ASTContext &C) override;
};

} // namespace polyregion::polystl