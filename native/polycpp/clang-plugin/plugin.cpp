
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"

#include "aspartame/all.hpp"

#include "polyfront/options_backend.hpp"
#include "polyregion/env_keys.h"

#include "ast.h"
#include "rewriter.h"

#ifdef POLYREGION_FUSED_DRIVER
// Fused build: polyreflect's plugin entry is statically linked; invoke it directly and feed its
// callbacks to CodeGenOpts.PassBuilderCallbacks (BackendUtil.cpp consumes them before the pipeline opens).
extern "C" llvm::PassPluginLibraryInfo llvmGetPassPluginInfo();
#endif

using namespace aspartame;
using namespace polyregion;
namespace {

class PolyCppFrontendAction final : public clang::PluginASTAction {

  polyfront::Options opts;

protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) override {
#ifdef POLYREGION_FUSED_DRIVER
    // XXX per-TU: CodeGenOpts is per-CompilerInstance.
    auto info = llvmGetPassPluginInfo();
    CI.getCodeGenOpts().PassBuilderCallbacks.push_back([info](llvm::PassBuilder &PB) { info.RegisterPassBuilderCallbacks(PB); });
#endif
    if (std::getenv(polyregion::env::PolycppNoRewrite)) return std::make_unique<clang::ASTConsumer>();
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
