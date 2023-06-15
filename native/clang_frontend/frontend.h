#pragma once

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

namespace polyregion::polystl {

class RewriteAndCompileAction : public clang::PluginASTAction {
private:
  clang::CompilerInstance *compilerInstance{};
  std::string fileName;
  clang::Rewriter rewriter;
  std::atomic_bool skipEndAction = false;

  std::unique_ptr<clang::FrontendAction> downstreamAction;
  std::function<std::unique_ptr<clang::ASTConsumer>(clang::Rewriter &, std::atomic_bool &)> createConsumer;

public:
  RewriteAndCompileAction(decltype(downstreamAction) downstreamAction, const decltype(createConsumer) &createConsumer);

protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef FileName) override;
  bool ParseArgs(clang::CompilerInstance const &, std::vector<std::string> const &) override;
  clang::PluginASTAction::ActionType getActionType() override;
  void EndSourceFileAction() override;
};

} // namespace polyregion::polystl