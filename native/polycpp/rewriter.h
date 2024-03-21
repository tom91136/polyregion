#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Rewrite/Core/Rewriter.h>

#include "options.h"

namespace polyregion::polystl {
  class OffloadRewriteConsumer : public clang::ASTConsumer {
  clang::DiagnosticsEngine &diag;
  StdParOptions opts;
public:
  OffloadRewriteConsumer(clang::DiagnosticsEngine &diag, const StdParOptions &opts);
  void HandleTranslationUnit(clang::ASTContext &context) override;
};

} // namespace polyregion::polystl