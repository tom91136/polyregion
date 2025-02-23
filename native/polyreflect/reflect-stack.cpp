#include <unordered_set>

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"

#include "aspartame/all.hpp"
#include "reflect-rt/rt_reflect.hpp"

#include "polyregion/llvm_ir.hpp"
#include "reflect-stack.h"

using namespace aspartame;
using namespace polyregion;

static bool runSplice(llvm::Module &M, const bool verbose) {
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

  std::unordered_set<llvm::Function *> Protected;
  llvm_shared::findFunctionsWithStringAnnotations(M, [&](llvm::Function *F, llvm::StringRef Annotation) {
    if (F && Annotation == "__rt_protect") Protected.emplace(F);
  });

  for (llvm::Function &F : M) {
    if (F.isDeclaration()) continue;
    if (Protected.count(&F) > 0) {
      if (verbose) llvm::errs() << "[ReflectStackPass] Skipping protected function " << F.getName() << "\n";
      continue;
    }

    llvm::DILocation *zeroDebugLoc{};
    if (const auto SP = F.getSubprogram()) {
      zeroDebugLoc = llvm::DILocation::get(C, 0, 0, SP);
    }

    if (verbose) llvm::errs() << "[ReflectStackPass] In " << F.getName() << "\n";
    for (llvm::BasicBlock &BB : F) {
      std::vector<std::function<void(llvm::IRBuilder<> &)>> Functions;
      for (llvm::Instruction &I : BB) {
        if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I)) {
          const auto Called = CB->getCalledFunction();
          if (!Called) continue;
          if (CB->getIntrinsicID() == llvm::Intrinsic::lifetime_start) {
            if (verbose) llvm::errs() << "[ReflectStackPass]   Inserted record ` " << *CB << "`\n";
            Functions.emplace_back([RecordFn, CB, zeroDebugLoc](llvm::IRBuilder<> &B) {
              B.SetInsertPoint(CB->getNextNode()); // after start
              auto alloc = B.getInt8(to_integral(rt_reflect::Type::StackAlloc));
              const auto Call = B.CreateCall(RecordFn, {CB->getArgOperand(1), CB->getArgOperand(0), alloc});
              if (zeroDebugLoc) Call->setDebugLoc(zeroDebugLoc);
            });
          }
          if (CB->getIntrinsicID() == llvm::Intrinsic::lifetime_end) {
            if (verbose) llvm::errs() << "[ReflectStackPass]   Inserted release ` " << *CB << "`\n";
            Functions.emplace_back([ReleaseFn, CB, zeroDebugLoc](llvm::IRBuilder<> &B) {
              B.SetInsertPoint(CB); // before end
              auto dealloc = B.getInt8(to_integral(rt_reflect::Type::StackFree));
              const auto Call = B.CreateCall(ReleaseFn, {CB->getArgOperand(1), dealloc});
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

polyreflect::ReflectStackPass::ReflectStackPass(const bool verbose) : verbose(verbose) {}
llvm::PreservedAnalyses polyreflect::ReflectStackPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  if (runSplice(M, verbose)) return llvm::PreservedAnalyses::none();
  return llvm::PreservedAnalyses::all();
}