#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Rewrite/Core/Rewriter.h>
namespace polyregion::polystl {
class OutlineConsumer : public clang::ASTConsumer {
private:
  clang::Rewriter &rewriter;
  std::atomic_bool &error;
//  std::unordered_map<int64_t, std::pair<clang::FunctionDecl *, clang::CXXRecordDecl *>> drain{};

public:
  OutlineConsumer(clang::Rewriter &rewriter, std::atomic_bool &error);
//  bool HandleTopLevelDecl(clang::DeclGroupRef DG) override;
  void HandleTranslationUnit(clang::ASTContext &context) override;
};

} // namespace polyregion::clang_frontend