#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "aspartame/all.hpp"
#include "magic_enum.hpp"
#include "polyregion/export.h"

#include "interpose.h"
#include "polyregion/llvm_ir.hpp"
#include "reflect-mem.h"
#include "reflect-stack.h"

using namespace aspartame;

namespace {

llvm::cl::opt<bool> verboseOpt("polyreflect-verbose", llvm::cl::init(false), //
                               llvm::cl::desc("Verbose output"), llvm::cl::Optional);
llvm::cl::opt<std::string> earlyPassesOpt("polyreflect-early", llvm::cl::init(""), //
                                          llvm::cl::desc("Early (StartEP) passes to run"), llvm::cl::Optional);
llvm::cl::opt<std::string> latePassesOpt("polyreflect-late", llvm::cl::init(""), //
                                         llvm::cl::desc("Late (LastEP) passes to run"), llvm::cl::Optional);

} // namespace

namespace polyregion::polyreflect {
class ProtectRTPass : public llvm::PassInfoMixin<ProtectRTPass> {
  bool verbose;

public:
  explicit ProtectRTPass(bool verbose) : verbose(verbose) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    llvm_shared::findFunctionsWithStringAnnotations(M, [&](llvm::Function *F, llvm::StringRef Annotation) {
      if (F && Annotation == "__rt_protect") {
        if (verbose) llvm::errs() << "[ProtectRTPass] LinkOnceODRLinkage for " << F->getName().str() << "\n";
        F->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
      }
    });
    return llvm::PreservedAnalyses::none();
  }
};
} // namespace polyregion::polyreflect

enum class PolyreflectPass : uint8_t { ProtectRT, Interpose, ReflectStack, ReflectMem };

extern "C" POLYREGION_EXPORT LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {

  const static auto parsePasses = [](const std::string &s) -> std::vector<PolyreflectPass> {
    if (s.empty()) return {};
    return s ^ split("+")           //
           ^ flat_map([](auto &v) { //
               auto parsed = magic_enum::enum_cast<PolyreflectPass>(v, magic_enum::case_insensitive);
               if (!parsed) llvm::errs() << "[PolyReflect] Unknown pass name `" << v << '`, ignoring...\n';
               return parsed ^ to_vector();
             }) //
           ^ distinct();
  };

  const static auto earlyPasses = parsePasses(earlyPassesOpt);
  const static auto latePasses = parsePasses(latePassesOpt);

  const static auto addPasses = [](llvm::ModulePassManager &MPM, const std::vector<PolyreflectPass> &passes) {
    for (const auto pass : passes) {
      if (verboseOpt) {
        llvm::errs() << "[PolyReflect] Add pass `" << magic_enum::enum_name(pass) << "`\n";
      }
      switch (pass) {
        case PolyreflectPass::ProtectRT: MPM.addPass(polyregion::polyreflect::ProtectRTPass(verboseOpt)); break;
        case PolyreflectPass::Interpose: MPM.addPass(polyregion::polyreflect::InterposePass(verboseOpt)); break;
        case PolyreflectPass::ReflectStack: MPM.addPass(polyregion::polyreflect::ReflectStackPass(verboseOpt)); break;
        case PolyreflectPass::ReflectMem: MPM.addPass(polyregion::polyreflect::ReflectMemPass(verboseOpt)); break;
        default: break;
      }
    }
  };

  return {LLVM_PLUGIN_API_VERSION, "polyreflect", LLVM_VERSION_STRING, [](llvm::PassBuilder &PB) {
            PB.registerOptimizerEarlyEPCallback([&](
#if LLVM_VERSION_MAJOR >= 20
                                                    llvm::ModulePassManager &MPM, llvm::OptimizationLevel, llvm::ThinOrFullLTOPhase
#else
                                                      llvm::ModulePassManager &MPM, llvm::OptimizationLevel
#endif
                                                ) { addPasses(MPM, earlyPasses); });
            PB.registerOptimizerLastEPCallback([&](
#if LLVM_VERSION_MAJOR >= 20
                                                   llvm::ModulePassManager &MPM, llvm::OptimizationLevel, llvm::ThinOrFullLTOPhase
#else
                                                   llvm::ModulePassManager &MPM, llvm::OptimizationLevel
#endif
                                               ) { addPasses(MPM, latePasses); });
            PB.registerFullLinkTimeOptimizationEarlyEPCallback(
                [&](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) { addPasses(MPM, earlyPasses); });
            PB.registerFullLinkTimeOptimizationLastEPCallback(
                [&](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) { addPasses(MPM, latePasses); });
          }};
}