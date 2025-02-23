#pragma once

#include "llvm/Passes/PassBuilder.h"

namespace polyregion::polyreflect {

class ReflectMemPass : public llvm::PassInfoMixin<ReflectMemPass> {
public:
  bool verbose;

  explicit ReflectMemPass(bool verbose);
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

} // namespace polyregion::polyreflect