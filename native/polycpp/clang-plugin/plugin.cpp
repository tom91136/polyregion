#include <iostream>

#include "plugin.h"
#include "rewriter.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "aspartame/all.hpp"

using namespace aspartame;

namespace {

class PolyCppFrontendAction : public clang::PluginASTAction {

  polyregion::polystl::Options opts;

protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) override {
    return std::make_unique<polyregion::polystl::OffloadRewriteConsumer>(CI.getDiagnostics(), opts);
  }

  bool ParseArgs(const clang::CompilerInstance &CI, const std::vector<std::string> &args) override {
    auto parseSuffix = [&](const std::string &arg, const std::string &prefix) -> std::optional<std::string> {
      if (arg ^ starts_with(prefix)) return arg.substr(prefix.size());
      return {};
    };
    auto &diag = CI.getDiagnostics();
    for (auto arg : args) {
      if (auto exe = parseSuffix(arg, "exe=")) {
        opts.executable = *exe;
        continue;
      }
      if (auto verbose = parseSuffix(arg, "verbose=")) {
        opts.verbose = *verbose == "1";
        continue;
      }
      if (auto targets = parseSuffix(arg, "targets=")) {
        // archA@featureA:archB@featureB:...archN@featureN
        for (auto &rawArchAndFeaturesList : *targets ^ split(':')) {
          auto archAndFeatures = rawArchAndFeaturesList ^ split('@');
          if (archAndFeatures.size() != 2) {
            diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                             "Missing or invalid placement of arch and feature separator '@' in %0"))
                << rawArchAndFeaturesList;
          }
          if (auto t = polyregion::compiletime::parseTarget(archAndFeatures[0]); t) {
            for (auto &feature : archAndFeatures[1] ^ split(','))
              opts.targets.emplace_back(*t, feature);
          } else {
            diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Error, "Unknown arch %0")) << archAndFeatures[0];
          }
        }
        continue;
      }
    }
    return true;
  }

  ActionType getActionType() override { return PluginASTAction::ActionType::CmdlineBeforeMainAction; }
};
} // namespace

[[maybe_unused]] static clang::FrontendPluginRegistry::Add<PolyCppFrontendAction> PolyCppClangPlugin("polycpp", "");
