#include "reflect-stack.h"

#include <unordered_set>

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"

#include "aspartame/all.hpp"

#include "polyregion/conventions.h"
#include "polyregion/llvm_ir.hpp"

#include "reflect-rt/rt_reflect.hpp"

using namespace aspartame;
using namespace polyregion;

static bool runSplice(llvm::Module &M, const bool verbose) {
  auto &C = M.getContext();
  // XXX getOrInsertFunction: bodies live in polystl-static / polydco-static.
  auto *i8Ty = llvm::Type::getInt8Ty(C);
  // XXX size arg is size_t; track ptr width, else aarch32 mints i64 into i32 sig
  auto *sizeTy = llvm::Type::getIntNTy(C, M.getDataLayout().getPointerSizeInBits());
  auto *voidTy = llvm::Type::getVoidTy(C);
  auto *ptrTy = llvm::PointerType::get(C, 0);
  auto *recordTy = llvm::FunctionType::get(voidTy, {ptrTy, sizeTy, i8Ty}, false);
  auto *releaseTy = llvm::FunctionType::get(voidTy, {ptrTy, i8Ty}, false);
  auto RecordFn = llvm::cast<llvm::Function>(M.getOrInsertFunction("_rt_record", recordTy).getCallee());
  auto ReleaseFn = llvm::cast<llvm::Function>(M.getOrInsertFunction("_rt_release", releaseTy).getCallee());

  std::unordered_set<llvm::Function *> Protected;
  llvm_shared::findFunctionsWithStringAnnotations(M, [&](llvm::Function *F, llvm::StringRef Annotation) {
    if (F &&                                                //
        (Annotation == POLYREFLECT_RT_PROTECT_ANNOTATION || //
         Annotation == POLYREFLECT_RT_ODR_ANNOTATION))
      Protected.emplace(F);
  });
  for (llvm::Function &F : M)
    if (F.hasFnAttribute(POLYREFLECT_RT_PROTECT_ANNOTATION)) Protected.emplace(&F);
  // XXX Transitive close: any function reachable from a protected one is also protected, else
  // the alloca walk below instruments stdlib helpers (e.g. std::atomic::load) inlined or called
  // from _rt_record's body and the inserted _rt_record calls recurse.
  std::vector<llvm::Function *> WorkList(Protected.begin(), Protected.end());
  while (!WorkList.empty()) {
    llvm::Function *F = WorkList.back();
    WorkList.pop_back();
    if (F->isDeclaration()) continue;
    for (llvm::BasicBlock &BB : *F) {
      for (llvm::Instruction &I : BB) {
        const auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
        if (!CB || CB->getIntrinsicID() != llvm::Intrinsic::not_intrinsic) continue;
        if (auto *Callee = CB->getCalledFunction()) {
          if (!Callee->isDeclaration() && Protected.emplace(Callee).second) WorkList.push_back(Callee);
        } else if (verbose) {
          // XXX Indirect call - we can't statically prove the callee is safe. The alloca walk
          // proceeds anyway; if recursion shows up at runtime, the indirect callee needs an
          // explicit polyreflect-rt-protect annotation.
          llvm::errs() << "[ReflectStackPass] Indirect call in protected function " << F->getName()
                       << "; unable to extend Protected set transitively\n";
        }
      }
    }
  }

  for (llvm::Function &F : M) {
    if (F.isDeclaration()) continue;
    if (Protected.count(&F) > 0) {
      if (verbose) llvm::errs() << "[ReflectStackPass] Skipping protected function " << F.getName() << "\n";
      continue;
    }

    std::vector<llvm::AllocaInst *> Allocas;
    for (llvm::Instruction &I : F.getEntryBlock()) {
      if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) Allocas.push_back(AI);
    }
    if (Allocas.empty()) continue;

    llvm::DILocation *zeroDebugLoc{};
    if (const auto SP = F.getSubprogram()) zeroDebugLoc = llvm::DILocation::get(C, 0, 0, SP);

    if (verbose) llvm::errs() << "[ReflectStackPass] In " << F.getName() << " (" << Allocas.size() << " allocas)\n";

    // XXX Anchor record/release on the alloca and on each return, not on llvm.lifetime_*. clang
    // only emits lifetime intrinsics at -O1+, and an alloca-scoped record is correct (just less
    // precise) for any opt level - polyreflect only needs the pointer to be tracked while the
    // kernel can observe it, which the function-scoped window always covers.
    for (auto *AI : Allocas) {
      llvm::IRBuilder B(AI->getNextNode());
      const auto sizeBytes = AI->getAllocationSize(M.getDataLayout()).value_or(llvm::TypeSize::getFixed(0)).getFixedValue();
      if (sizeBytes == 1) B.CreateMemSet(AI, B.getInt8(0), B.getInt64(1), AI->getAlign());
      const auto Call =
          B.CreateCall(RecordFn, {AI, llvm::ConstantInt::get(sizeTy, sizeBytes), B.getInt8(to_integral(rt_reflect::Type::StackAlloc))});
      if (zeroDebugLoc) Call->setDebugLoc(zeroDebugLoc);
    }
    for (llvm::BasicBlock &BB : F) {
      if (auto *Ret = llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator())) {
        llvm::IRBuilder B(Ret);
        const auto dealloc = B.getInt8(to_integral(rt_reflect::Type::StackFree));
        for (auto *AI : Allocas) {
          const auto Call = B.CreateCall(ReleaseFn, {AI, dealloc});
          if (zeroDebugLoc) Call->setDebugLoc(zeroDebugLoc);
        }
      }
    }
  }
  return true;
}

polyreflect::ReflectStackPass::ReflectStackPass(const bool verbose) : verbose(verbose) {}
llvm::PreservedAnalyses polyreflect::ReflectStackPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  if (runSplice(M, verbose)) return llvm::PreservedAnalyses::none();
  return llvm::PreservedAnalyses::all();
}