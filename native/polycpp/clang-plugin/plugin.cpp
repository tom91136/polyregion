
#include <memory>
#include <string>
#include <vector>

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

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

struct CapturedLinkDiagnostic {
  llvm::DiagnosticSeverity severity;
  std::string message;
};

class LinkDiagnosticHandler final : public llvm::DiagnosticHandler {
  llvm::DiagnosticHandler *previous;
  std::vector<CapturedLinkDiagnostic> &diagnostics;

public:
  LinkDiagnosticHandler(llvm::DiagnosticHandler *previous, std::vector<CapturedLinkDiagnostic> &diagnostics)
      : previous(previous), diagnostics(diagnostics) {}

  bool handleDiagnostics(const llvm::DiagnosticInfo &diagnostic) override {
    if (diagnostic.getKind() != llvm::DK_Linker) {
      if (previous && diagnostic.getSeverity() == llvm::DS_Error) previous->HasErrors = true;
      return previous && previous->handleDiagnostics(diagnostic);
    }
    if (previous && diagnostic.getSeverity() == llvm::DS_Error) previous->HasErrors = true;
    std::string message;
    llvm::raw_string_ostream stream(message);
    llvm::DiagnosticPrinterRawOStream printer(stream);
    diagnostic.print(printer);
    diagnostics.emplace_back(diagnostic.getSeverity(), std::move(message));
    return true;
  }
};

bool linkDriverModule(llvm::Module &module, std::unique_ptr<llvm::Module> driver, clang::DiagnosticsEngine &clangDiagnostics) {
  auto &context = module.getContext();
  auto previous = context.getDiagnosticHandler();
  const auto identifier = driver->getModuleIdentifier();
  std::vector<CapturedLinkDiagnostic> diagnostics;
  context.setDiagnosticHandler(std::make_unique<LinkDiagnosticHandler>(previous.get(), diagnostics));
  const bool failed = llvm::Linker(module).linkInModule(std::move(driver));
  context.setDiagnosticHandler(std::move(previous));
  diagnostics | for_each([&](const auto &diagnostic) {
    unsigned id;
    switch (diagnostic.severity) {
      case llvm::DS_Error: id = clang::diag::err_fe_linking_module; break;
      case llvm::DS_Warning: id = clang::diag::warn_fe_linking_module; break;
      case llvm::DS_Note: id = clang::diag::note_fe_linking_module; break;
      case llvm::DS_Remark: return;
    }
    clangDiagnostics.Report(id) << identifier << diagnostic.message;
  });
  return failed;
}

class LinkDriverBitcodePass final : public llvm::PassInfoMixin<LinkDriverBitcodePass> {
  std::shared_ptr<polystl::DriverBitcode> bitcode;
  clang::DiagnosticsEngine &diagnostics;

public:
  LinkDriverBitcodePass(std::shared_ptr<polystl::DriverBitcode> bitcode, clang::DiagnosticsEngine &diagnostics)
      : bitcode(std::move(bitcode)), diagnostics(diagnostics) {}

  llvm::PreservedAnalyses run(llvm::Module &module, llvm::ModuleAnalysisManager &) {
    for (const auto &bytes : *bitcode) {
      const auto data = llvm::StringRef(reinterpret_cast<const char *>(bytes.data()), bytes.size());
      auto parsed = llvm::parseBitcodeFile(llvm::MemoryBufferRef(data, "polyregion-interface-driver"), module.getContext());
      if (!parsed) {
        module.getContext().emitError(llvm::toString(parsed.takeError()));
        return llvm::PreservedAnalyses::none();
      }
      auto driver = std::move(*parsed);
      const llvm::Triple sourceTriple(driver->getTargetTriple()), targetTriple(module.getTargetTriple());
      if (!polyfront::objectTargetsCompatible(sourceTriple, targetTriple)
          || !polyfront::objectLayoutsCompatible(*driver, module.getDataLayout())) {
        module.getContext().emitError("PolyAST interface driver target is incompatible with the translation unit");
        return llvm::PreservedAnalyses::none();
      }
      driver->setTargetTriple(module.getTargetTriple());
      driver->setDataLayout(module.getDataLayout());
      if (linkDriverModule(module, std::move(driver), diagnostics)) {
        module.getContext().emitError("cannot link PolyAST interface driver");
        return llvm::PreservedAnalyses::none();
      }
    }
    return bitcode->empty() ? llvm::PreservedAnalyses::all() : llvm::PreservedAnalyses::none();
  }
};

class PolyCppFrontendAction final : public clang::PluginASTAction {

  polyfront::Options opts;

protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) override {
    auto driverBitcode = std::make_shared<polystl::DriverBitcode>();
    auto &diagnostics = CI.getDiagnostics();
    CI.getCodeGenOpts().PassBuilderCallbacks.push_back([driverBitcode, &diagnostics](llvm::PassBuilder &PB) {
      PB.registerPipelineStartEPCallback([driverBitcode, &diagnostics](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
        MPM.addPass(LinkDriverBitcodePass(driverBitcode, diagnostics));
      });
    });
#ifdef POLYREGION_FUSED_DRIVER
    // XXX per-TU: CodeGenOpts is per-CompilerInstance.
    auto info = llvmGetPassPluginInfo();
    CI.getCodeGenOpts().PassBuilderCallbacks.push_back([info](llvm::PassBuilder &PB) { info.RegisterPassBuilderCallbacks(PB); });
#endif
    if (std::getenv(polyregion::env::PolycppNoRewrite)) return std::make_unique<clang::ASTConsumer>();
    return std::make_unique<polystl::OffloadRewriteConsumer>(CI, opts, std::move(driverBitcode));
  }

  bool ParseArgs(const clang::CompilerInstance &CI, const std::vector<std::string> &args) override {
    polyfront::Options::parseArgs(args) //
        ^ foreach_total([&](const polyfront::Options &x) { opts = x; },
                        [&](const std::vector<std::string> &errors) {
                          auto &diag = CI.getDiagnostics();
                          errors | for_each([&](const auto &error) {
                            diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Error, "%0")) << error;
                          });
                        });
    return true;
  }

  ActionType getActionType() override { return CmdlineBeforeMainAction; }
};
} // namespace

[[maybe_unused]] static clang::FrontendPluginRegistry::Add<PolyCppFrontendAction> PolyCppClangPlugin("polycpp", "");
