#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Rewrite/Core/Rewriter.h>
namespace polyregion::polystl {
  class OffloadRewriteConsumer : public clang::ASTConsumer {
  clang::DiagnosticsEngine &diag;
public:
  explicit OffloadRewriteConsumer(clang::DiagnosticsEngine &diag);
  void HandleTranslationUnit(clang::ASTContext &context) override;
};

} // namespace polyregion::polystl