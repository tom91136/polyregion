#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "fmt/core.h"

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

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

#include "ast.h"
#include "clang_utils.h"
#include "compiler.h"
#include "frontend.h"
#include "remapper.h"
#include "rewriter.h"
namespace {

using namespace polyregion::polystl;


struct PluginRewriteAndCompileAction : public ModifyASTAndEmitObjAction {
  PluginRewriteAndCompileAction()
      : ModifyASTAndEmitObjAction(std::make_unique<clang::EmitObjAction>(),
                                [](clang::Rewriter &r, std::atomic_bool &error) { return std::make_unique<OffloadRewriteConsumer>(r, error); }) {}
};

} // end namespace

static clang::FrontendPluginRegistry::Add<PluginRewriteAndCompileAction> X("fire", "create CLI from functions or classes");
