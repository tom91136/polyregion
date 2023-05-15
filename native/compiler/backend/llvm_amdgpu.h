#pragma once

#include "llvm.h"

namespace polyregion::backend {

class AMDGPULLVMTargetHandler : public LLVMTargetSpecificHandler {
  void witnessEntry(llvm::LLVMContext &ctx, llvm::Module &mod, llvm::Function &fn) override;
};

} // namespace polyregion::backend