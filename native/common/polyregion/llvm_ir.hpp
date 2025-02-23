#pragma once

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

namespace polyregion::llvm_shared {

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

template <typename Handler> void findValuesWithStringAnnotations(llvm::Module &M, Handler &&f) {
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (const auto CI = dyn_cast<llvm::CallInst>(&I)) {
          if (CI->getIntrinsicID() != llvm::Intrinsic::var_annotation) continue;
          const auto V = CI->getArgOperand(0)->stripPointerCasts();
          const auto annoArg = dyn_cast<llvm::Constant>(CI->getArgOperand(1));
          if (!annoArg) continue;
          const auto annoStrGV = llvm::dyn_cast<llvm::GlobalVariable>(annoArg->stripPointerCasts());
          if (!annoStrGV || !annoStrGV->hasInitializer()) continue;
          if (const auto *CDA = llvm::dyn_cast<llvm::ConstantDataArray>(annoStrGV->getInitializer())) {
            f(F, V, CDA->getAsString());
          }
        }
      }
    }
  }
}

} // namespace polyregion::llvm_shared
