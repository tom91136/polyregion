#pragma once

#include "llvm/Passes/PassBuilder.h"

namespace polyregion::polyreflect {

class ReflectStackPass : public llvm::PassInfoMixin<ReflectStackPass> {
  bool verbose;

public:
  explicit ReflectStackPass(bool verbose);
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

} // namespace polyregion::polyreflect