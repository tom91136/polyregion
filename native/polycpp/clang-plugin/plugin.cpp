#include <iostream>

#include "rewriter.h"

#include "aspartame/all.hpp"
#include "polyfront/options_backend.hpp"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace aspartame;
using namespace polyregion;
namespace {

class PolyCppFrontendAction final : public clang::PluginASTAction {

  polyfront::Options opts;

protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) override {
    return std::make_unique<polystl::OffloadRewriteConsumer>(CI, opts);
  }

  bool ParseArgs(const clang::CompilerInstance &CI, const std::vector<std::string> &args) override {
    polyfront::Options::parseArgs(args) //
        ^ foreach_total([&](const polyfront::Options &x) { opts = x; },
                        [&](const std::vector<std::string> &errors) {
                          auto &diag = CI.getDiagnostics();
                          for (auto error : errors)
                            diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Error, "%0")) << error;
                        });
    return true;
  }

  ActionType getActionType() override { return CmdlineBeforeMainAction; }
};
} // namespace

[[maybe_unused]] static clang::FrontendPluginRegistry::Add<PolyCppFrontendAction> PolyCppClangPlugin("polycpp", "");
