#pragma once

#include "plugin.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/Diagnostic.h"

namespace polyregion::polystl {

class OffloadRewriteConsumer : public clang::ASTConsumer {
  clang::DiagnosticsEngine &diag;
  Options opts;

public:
  OffloadRewriteConsumer(clang::DiagnosticsEngine &diag, const Options &opts);
  void HandleTranslationUnit(clang::ASTContext &context) override;
};

} // namespace polyregion::polystl