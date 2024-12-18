#include <filesystem>
#include <unordered_set>

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "polyregion/export.h"
#include "ptr-reflect-rt/rt_reflect.hpp"

namespace {

// XXX Stolen from
// https://github.com/AdaptiveCpp/AdaptiveCpp/blob/061e2d6ffe1084021d99f22ac1f16e28c6dab899/include/hipSYCL/compiler/cbs/IRUtils.hpp#L183
template <class T> T *getValueOneLevel(llvm::Constant *V, unsigned idx = 0) {
  // opaque ptr
  if (auto *R = llvm::dyn_cast<T>(V)) return R;

  // typed ptr -> look through bitcast
  if (V->getNumOperands() == 0) return nullptr;
  return llvm::dyn_cast<T>(V->getOperand(idx));
}

// XXX Stolen from
// https://github.com/AdaptiveCpp/AdaptiveCpp/blob/061e2d6ffe1084021d99f22ac1f16e28c6dab899/include/hipSYCL/compiler/cbs/IRUtils.hpp#L195
template <class Handler> void findFunctionsWithStringAnnotationsWithArg(llvm::Module &M, Handler &&f) {
  for (auto &I : M.globals()) {
    if (I.getName() == "llvm.global.annotations") {
      auto *CA = llvm::dyn_cast<llvm::ConstantArray>(I.getOperand(0));
      for (auto *OI = CA->op_begin(); OI != CA->op_end(); ++OI) {
        if (auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(OI->get()); CS && CS->getNumOperands() >= 2)
          if (auto *F = getValueOneLevel<llvm::Function>(CS->getOperand(0)))
            if (auto *AnnotationGL = getValueOneLevel<llvm::GlobalVariable>(CS->getOperand(1)))
              if (auto *Initializer = llvm::dyn_cast<llvm::ConstantDataArray>(AnnotationGL->getInitializer())) {
                llvm::StringRef Annotation = Initializer->getAsCString();
                f(F, Annotation, CS->getNumOperands() > 3 ? CS->getOperand(4) : nullptr);
              }
      }
    }
  }
}

// XXX Stolen from
// https://github.com/AdaptiveCpp/AdaptiveCpp/blob/061e2d6ffe1084021d99f22ac1f16e28c6dab899/include/hipSYCL/compiler/cbs/IRUtils.hpp#L215
template <class Handler> void findFunctionsWithStringAnnotations(llvm::Module &M, Handler &&f) {
  findFunctionsWithStringAnnotationsWithArg(M, [&f](llvm::Function *F, llvm::StringRef Annotation, llvm::Value *) { f(F, Annotation); });
}

bool runSplice(llvm::Module &M) {
  // M.print(llvm::errs(), nullptr);
  auto &C = M.getContext();
  auto RecordFn = M.getFunction("_rt_record");
  auto ReleaseFn = M.getFunction("_rt_release");
  if (!RecordFn) {
    llvm::errs() << "[RecordStackPass] _rt_record not found, giving up\n";
    return false;
  }
  if (!ReleaseFn) {
    llvm::errs() << "[RecordStackPass] _rt_release not found, giving up\n";
    return false;
  }

  std::unordered_set<llvm::Function *> ProtectedFunctions;
  findFunctionsWithStringAnnotations(M, [&](llvm::Function *F, llvm::StringRef Annotation) {
    if (F && Annotation == "__rt_protect") ProtectedFunctions.emplace(F);
  });

  for (llvm::Function &F : M) {
    if (F.isDeclaration()) continue;
    if (ProtectedFunctions.count(&F) > 0) continue;

    llvm::DILocation *zeroDebugLoc{};
    if (auto SP = F.getSubprogram()) {
      zeroDebugLoc = llvm::DILocation::get(C, 0, 0, SP);
    }

    for (llvm::BasicBlock &BB : F) {
      std::vector<std::function<void(llvm::IRBuilder<> &)>> Functions;
      for (llvm::Instruction &I : BB) {
        if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I)) {
          auto F = CB->getCalledFunction();
          if (!F) continue;
          const auto name = F->getName().str();
          const auto log = [&]() {};
          if (CB->getIntrinsicID() == llvm::Intrinsic::lifetime_start) {
            log();
            Functions.emplace_back([RecordFn, CB, zeroDebugLoc](llvm::IRBuilder<> &B) {
              B.SetInsertPoint(CB->getNextNode()); // after start
              auto alloc = B.getInt8(to_integral(ptr_reflect::_rt_Type::StackAlloc));
              const auto Call = B.CreateCall(RecordFn, {CB->getArgOperand(1), CB->getArgOperand(0), alloc});
              if (zeroDebugLoc) Call->setDebugLoc(zeroDebugLoc);
            });
          }
          if (CB->getIntrinsicID() == llvm::Intrinsic::lifetime_end) {
            log();
            Functions.emplace_back([ReleaseFn, CB, zeroDebugLoc](llvm::IRBuilder<> &B) {
              B.SetInsertPoint(CB); // before end
              auto dealloc = B.getInt8(to_integral(ptr_reflect::_rt_Type::StackFree));
              auto Call = B.CreateCall(ReleaseFn, {CB->getArgOperand(1), dealloc});
              if (zeroDebugLoc) Call->setDebugLoc(zeroDebugLoc);
            });
          }
        }
      }
      llvm::IRBuilder B(&BB);
      for (auto f : Functions)
        f(B);
    }
  }
  return true;
}

} // namespace

class RecordStackPass : public llvm::PassInfoMixin<RecordStackPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    if (runSplice(M)) return llvm::PreservedAnalyses::none();
    return llvm::PreservedAnalyses::all();
  }
};

class ProtectRTPass : public llvm::PassInfoMixin<ProtectRTPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &) {

    findFunctionsWithStringAnnotations(M, [&](llvm::Function *F, llvm::StringRef Annotation) {
      if (F && Annotation == "__rt_protect") F->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
    });

    return llvm::PreservedAnalyses::none();
  }
};

extern "C" POLYREGION_EXPORT LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "ptr-reflect", LLVM_VERSION_STRING, [](llvm::PassBuilder &PB) {
        PB.registerPipelineStartEPCallback([&](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) { MPM.addPass(ProtectRTPass()); });
        PB.registerOptimizerLastEPCallback([&](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) { MPM.addPass(RecordStackPass()); });
        PB.registerFullLinkTimeOptimizationLastEPCallback(
            [&](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) { MPM.addPass(RecordStackPass()); });
      }};
}