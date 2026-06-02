#pragma once

#include "llvm/Passes/PassBuilder.h"

namespace polyregion::polyreflect {

// MemKind::Interpose - redirect malloc/operator new to USM allocators (polyrt_usm_*).
class InterposePass : public llvm::PassInfoMixin<InterposePass> {
public:
  bool verbose;

  explicit InterposePass(bool verbose);
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

class RecordAllocPass : public llvm::PassInfoMixin<RecordAllocPass> {
public:
  bool verbose;

  explicit RecordAllocPass(bool verbose);
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

} // namespace polyregion::polyreflect