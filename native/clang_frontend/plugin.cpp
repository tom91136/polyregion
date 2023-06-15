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
#include "outliner.h"
#include "remapper.h"
namespace {

using namespace polyregion::polystl;

static std::optional<CompileResult> compileIt(Program &p) {
  auto data = nlohmann::json::to_msgpack(hashed_to_json(program_to_json(p)));

  //    llvm::sys::fs::createTemporaryFile("","", )

  llvm::SmallString<64> inputPath;
  auto inputCreateEC = llvm::sys::fs::createTemporaryFile("", "", inputPath);
  if (inputCreateEC) {
    llvm::errs() << "Failed to create temp input file: " << inputCreateEC.message() << "\n";
    return {};
  }

  llvm::SmallString<64> outputPath;
  auto outputCreateEC = llvm::sys::fs::createTemporaryFile("", "", outputPath);
  if (outputCreateEC) {
    llvm::errs() << "Failed to create temp output file: " << outputCreateEC.message() << "\n";
    return {};
  }

  std::error_code streamEC;
  llvm::raw_fd_ostream File(inputPath, streamEC, llvm::sys::fs::OF_None);
  if (streamEC) {
    llvm::errs() << "Failed to open file: " << streamEC.message() << "\n";
    return {};
  }
  File.write(reinterpret_cast<const char *>(data.data()), data.size());
  File.flush();
  llvm::outs() << "Wrote " << inputPath.str() << " \n";

  int code = llvm::sys::ExecuteAndWait(                                                              //
      "/home/tom/polyregion/native/cmake-build-debug-clang/compiler/polyc",                          //
      {"polyc", inputPath.str(), "--out", outputPath.str(), "--target", "host", "--arch", "native"}, //
      {{}}                                                                                           //
  );

  auto BufferOrErr = llvm::MemoryBuffer::getFile(outputPath);

  if (auto Err = BufferOrErr.getError()) {
    llvm::errs() << llvm::errorCodeToError(Err) << "\n";
    return {};
  } else {

    return compileresult_from_json(nlohmann::json::from_msgpack((*BufferOrErr)->getBufferStart(), (*BufferOrErr)->getBufferEnd()));
  }
}

struct PluginRewriteAndCompileAction : public RewriteAndCompileAction {
  PluginRewriteAndCompileAction()
      : RewriteAndCompileAction(std::make_unique<clang::EmitObjAction>(),
                                [](clang::Rewriter &r, std::atomic_bool &error) { return std::make_unique<OutlineConsumer>(r, error); }) {}
};

} // end namespace

static clang::FrontendPluginRegistry::Add<PluginRewriteAndCompileAction> X("fire", "create CLI from functions or classes");
