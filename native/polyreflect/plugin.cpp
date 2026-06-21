#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include "aspartame/all.hpp"

#include "polyregion/conventions.h"
#include "polyregion/export.h"
#include "polyregion/llvm_ir.hpp"
#include "polyregion/mirror_names.h"

#include "interpose.h"
#include "reflect-mem.h"
#include "reflect-stack.h"

using namespace aspartame;

namespace {

llvm::cl::opt<bool> verboseOpt(llvm::StringRef(polyregion::conventions::reflect::FlagVerbose), llvm::cl::init(false), //
                               llvm::cl::desc("Verbose output"), llvm::cl::Optional);
llvm::cl::opt<std::string> earlyPassesOpt(llvm::StringRef(polyregion::conventions::reflect::FlagEarly), llvm::cl::init(""), //
                                          llvm::cl::desc("Early (StartEP) passes to run"), llvm::cl::Optional);
llvm::cl::opt<std::string> latePassesOpt(llvm::StringRef(polyregion::conventions::reflect::FlagLate), llvm::cl::init(""), //
                                         llvm::cl::desc("Late (LastEP) passes to run"), llvm::cl::Optional);

} // namespace

namespace polyregion::polyreflect {
class ProtectRTPass : public llvm::PassInfoMixin<ProtectRTPass> {
  bool verbose;

public:
  explicit ProtectRTPass(bool verbose) : verbose(verbose) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    // XXX COFF needs a comdat for LinkOnceODR; ELF/Mach-O don't.
    const bool isCOFF = M.getTargetTriple().isOSBinFormatCOFF();
    llvm_shared::findFunctionsWithStringAnnotations(M, [&](llvm::Function *F, llvm::StringRef Annotation) {
      if (F && Annotation == POLYREFLECT_RT_ODR_ANNOTATION) {
        if (verbose) llvm::errs() << "[ProtectRTPass] LinkOnceODRLinkage for " << F->getName().str() << "\n";
        F->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
        if (isCOFF && !F->hasComdat()) F->setComdat(M.getOrInsertComdat(F->getName()));
      }
    });
    return llvm::PreservedAnalyses::none();
  }
};
} // namespace polyregion::polyreflect

namespace polyregion::polyreflect {
// links the embedded `polyregion_mirror_bc` global and wires prelude/postlude into each KernelBundle; no-op if absent
class LinkMirrorPass : public llvm::PassInfoMixin<LinkMirrorPass> {
  bool verbose;

public:
  explicit LinkMirrorPass(bool verbose) : verbose(verbose) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    llvm::SmallVector<llvm::GlobalVariable *, 2> bcGlobals;
    for (llvm::GlobalVariable &G : M.globals())
      if (G.hasInitializer() && G.getName().contains(conventions::reflect::MirrorBitcodeGlobal)) bcGlobals.push_back(&G);

    // re-applies at every EP: under `-g` a mid-pipeline GlobalOpt nulls the bundle fields after early wiring
    bool linked = false;
    if (!bcGlobals.empty()) {
      // clang's diagnostic handler asserts (`CurLinkModule must be set`) outside its own linking; swap it out
      auto &Ctx = M.getContext();
      auto prevHandler = Ctx.getDiagnosticHandler();
      Ctx.setDiagnosticHandler(std::make_unique<llvm::DiagnosticHandler>(), /*RespectFilters*/ false);

      for (auto *G : bcGlobals) {
        auto *init = llvm::dyn_cast<llvm::ConstantDataArray>(G->getInitializer());
        if (!init) continue;
        const llvm::StringRef bytes = init->getRawDataValues();
        auto buf = llvm::MemoryBuffer::getMemBuffer(bytes, "mirror.bc", /*RequiresNullTerminator*/ false);
        auto modOr = llvm::parseBitcodeFile(buf->getMemBufferRef(), M.getContext());
        if (!modOr) {
          llvm::errs() << "[LinkMirror] parseBitcodeFile failed: " << llvm::toString(modOr.takeError()) << "\n";
          continue;
        }
        if (llvm::Linker::linkModules(M, std::move(*modOr))) {
          llvm::errs() << "[LinkMirror] linkModules failed\n";
          continue;
        }
        if (verbose) llvm::errs() << "[LinkMirror] linked " << bytes.size() << " bytes from " << G->getName().str() << "\n";
        linked = true;
      }

      Ctx.setDiagnosticHandler(std::move(prevHandler));

      if (linked) {
        llvm::removeFromUsedLists(M, [](llvm::Constant *c) {
          auto *gv = llvm::dyn_cast<llvm::GlobalVariable>(c);
          return gv && gv->getName().contains(conventions::reflect::MirrorBitcodeGlobal);
        });
        for (auto *G : bcGlobals) {
          G->replaceAllUsesWith(llvm::UndefValue::get(G->getType()));
          G->eraseFromParent();
        }
        // pin prelude/postlude so they survive to OptimizerLastEP (compiler.used drops before codegen)
        for (llvm::Function &F : M.functions())
          if (F.getName().starts_with(std::string(conventions::reflect::MirrorPrelude) + "_") ||
              F.getName().starts_with(std::string(conventions::reflect::MirrorPostlude) + "_"))
            llvm::appendToCompilerUsed(M, {&F});
      }
    }

    const bool wired = wireBundles(M);
    return (linked || wired) ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
  }

  // wire each KernelBundle to its prelude/postlude via its mirrorId field; an empty id stays null
  bool wireBundles(llvm::Module &M) {
    const auto bundleCString = [](llvm::ConstantStruct *cs, unsigned i) -> std::optional<std::string> {
      if (cs->getNumOperands() <= i) return std::nullopt;
      auto *gv = llvm::dyn_cast<llvm::GlobalVariable>(cs->getOperand(i)->stripPointerCasts());
      if (!gv || !gv->hasInitializer()) return std::nullopt;
      auto *arr = llvm::dyn_cast<llvm::ConstantDataArray>(gv->getInitializer());
      if (!arr || !arr->isCString()) return std::nullopt;
      return arr->getAsCString().str();
    };
    bool wiredAny = false;
    // wire one KernelBundle struct; polycpp emits a bare ConstantStruct, polyfc a ConstantArray, both route here
    const auto wireStruct = [&](llvm::ConstantStruct *cs, llvm::StringRef gName) -> llvm::Constant * {
      auto *st = cs->getType();
      if (!st->hasName() || !st->getName().contains(conventions::KernelBundleType) || st->getNumElements() < 3) return cs;
      const unsigned n = st->getNumElements();
      const auto id = bundleCString(cs, n - 3); // mirrorId sits just before prelude/postlude
      if (!id || id->empty()) {
        if (verbose) llvm::errs() << "[LinkMirror] bundle " << gName.str() << " (" << st->getName().str() << "): no mirrorId\n";
        return cs;
      }
      auto *prelude = M.getFunction(mirror::preludeName(*id));
      auto *postlude = M.getFunction(mirror::postludeName(*id));
      if (!prelude || !postlude) {
        if (verbose)
          llvm::errs() << "[LinkMirror] bundle " << gName.str() << " id=" << *id << " preludeName=" << mirror::preludeName(*id)
                       << " prelude=" << (prelude ? "y" : "n") << " postlude=" << (postlude ? "y" : "n") << "\n";
        return cs;
      }
      llvm::SmallVector<llvm::Constant *, 16> ops;
      for (unsigned i = 0; i < n; ++i)
        ops.push_back(cs->getOperand(i));
      ops[n - 2] = llvm::ConstantExpr::getBitCast(prelude, st->getElementType(n - 2));
      ops[n - 1] = llvm::ConstantExpr::getBitCast(postlude, st->getElementType(n - 1));
      wiredAny = true;
      if (verbose) llvm::errs() << "[LinkMirror] wired prelude/postlude " << *id << " into " << gName.str() << "\n";
      return llvm::ConstantStruct::get(st, ops);
    };
    for (llvm::GlobalVariable &G : M.globals()) {
      if (!G.hasInitializer()) continue;
      auto *init = G.getInitializer();
      if (auto *cs = llvm::dyn_cast<llvm::ConstantStruct>(init)) {
        if (auto *wired = wireStruct(cs, G.getName()); wired != cs) G.setInitializer(wired);
      } else if (auto *ca = llvm::dyn_cast<llvm::ConstantArray>(init)) {
        auto *elemTy = llvm::dyn_cast<llvm::StructType>(ca->getType()->getElementType());
        if (!elemTy || !elemTy->hasName() || !elemTy->getName().contains(conventions::KernelBundleType)) continue;
        llvm::SmallVector<llvm::Constant *, 4> elems;
        bool changed = false;
        for (unsigned i = 0; i < ca->getNumOperands(); ++i) {
          auto *eltCs = llvm::dyn_cast<llvm::ConstantStruct>(ca->getOperand(i));
          auto *wired = eltCs ? wireStruct(eltCs, G.getName()) : ca->getOperand(i);
          changed |= wired != ca->getOperand(i);
          elems.push_back(wired);
        }
        if (changed) G.setInitializer(llvm::ConstantArray::get(ca->getType(), elems));
      }
    }
    return wiredAny;
  }
};
} // namespace polyregion::polyreflect

namespace {
// add the polyreflect pass named by its conventions identifier; false if the name is unrecognised
bool addReflectPass(llvm::ModulePassManager &MPM, llvm::StringRef name) {
  namespace conv = polyregion::conventions::reflect;
  using namespace polyregion::polyreflect;
  if (name == conv::PassLinkMirror) MPM.addPass(LinkMirrorPass(verboseOpt));
  else if (name == conv::PassProtectRt) MPM.addPass(ProtectRTPass(verboseOpt));
  else if (name == conv::PassInterpose) MPM.addPass(InterposePass(verboseOpt));
  else if (name == conv::PassRecordAlloc) MPM.addPass(RecordAllocPass(verboseOpt));
  else if (name == conv::PassStack) MPM.addPass(ReflectStackPass(verboseOpt));
  else if (name == conv::PassMem) MPM.addPass(ReflectMemPass(verboseOpt));
  else return false;
  if (verboseOpt) llvm::errs() << "[PolyReflect] Add pass `" << name << "`\n";
  return true;
}

// schedule a '+'-separated -polyreflect-early/-late spec (the conventions pass names)
void addReflectSpec(llvm::ModulePassManager &MPM, const std::string &spec) {
  for (const auto &name : spec ^ split("+") ^ distinct())
    if (!name.empty() && !addReflectPass(MPM, name)) llvm::errs() << "[PolyReflect] Unknown pass name `" << name << "`, ignoring...\n";
}
} // namespace

extern "C" POLYREGION_EXPORT LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "polyreflect", LLVM_VERSION_STRING, [](llvm::PassBuilder &PB) {
            // wire before the inliner constant-folds the `bundle.prelude` load into the offload wrapper
            PB.registerPipelineStartEPCallback([&](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
              MPM.addPass(polyregion::polyreflect::LinkMirrorPass(verboseOpt));
            });
            // no LinkMirrorPass at EarlyEP: PipelineStart already linked + erased the bc globals
            PB.registerOptimizerEarlyEPCallback([&](
#if LLVM_VERSION_MAJOR >= 20
                                                    llvm::ModulePassManager &MPM, llvm::OptimizationLevel, llvm::ThinOrFullLTOPhase
#else
                                                      llvm::ModulePassManager &MPM, llvm::OptimizationLevel
#endif
                                                ) { addReflectSpec(MPM, earlyPassesOpt); });
            PB.registerOptimizerLastEPCallback([&](
#if LLVM_VERSION_MAJOR >= 20
                                                   llvm::ModulePassManager &MPM, llvm::OptimizationLevel, llvm::ThinOrFullLTOPhase
#else
                                                   llvm::ModulePassManager &MPM, llvm::OptimizationLevel
#endif
                                               ) {
              // re-wire after GlobalOpt (which, under `-g`, nulls the bundle's mirror fields set at EarlyEP)
              MPM.addPass(polyregion::polyreflect::LinkMirrorPass(verboseOpt));
              addReflectSpec(MPM, latePassesOpt);
            });
            PB.registerFullLinkTimeOptimizationEarlyEPCallback([&](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
              MPM.addPass(polyregion::polyreflect::LinkMirrorPass(verboseOpt));
              addReflectSpec(MPM, earlyPassesOpt);
            });
            PB.registerFullLinkTimeOptimizationLastEPCallback([&](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
              MPM.addPass(polyregion::polyreflect::LinkMirrorPass(verboseOpt));
              addReflectSpec(MPM, latePassesOpt);
            });
            // XXX At -O0, LLD's LTO codegen short-circuits without building any optimisation
            // pipeline, so EP callbacks never fire. Register passes by name as well so they can
            // be inserted explicitly via -Wl,--lto-newpm-passes=... at link time.
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::ModulePassManager &MPM, llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  return addReflectPass(MPM, Name);
                });
          }};
}