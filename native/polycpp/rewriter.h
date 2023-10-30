#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Rewrite/Core/Rewriter.h>
namespace polyregion::polystl {
class OffloadRewriteConsumer : public clang::ASTConsumer {
private:
  //  std::unordered_map<int64_t, std::pair<clang::FunctionDecl *, clang::CXXRecordDecl *>> drain{};
public:
  OffloadRewriteConsumer();
  void HandleTranslationUnit(clang::ASTContext &context) override;
};

} // namespace polyregion::polystl