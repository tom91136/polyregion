#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Rewrite/Core/Rewriter.h>

#include "options.h"

namespace polyregion::polystl {
class OffloadRewriteConsumer : public clang::ASTConsumer {
  clang::DiagnosticsEngine &diag;
  DriverContext ctx;

public:
  OffloadRewriteConsumer(clang::DiagnosticsEngine &diag, const DriverContext &ctx);
  void HandleTranslationUnit(clang::ASTContext &context) override;
};

} // namespace polyregion::polystl