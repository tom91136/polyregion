#pragma once

#include "llvm/Passes/PassBuilder.h"

namespace polyregion::polyreflect {

class InterposePass : public llvm::PassInfoMixin<InterposePass> {
public:
  bool verbose;

  explicit InterposePass(bool verbose);
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

} // namespace polyregion::polyreflect