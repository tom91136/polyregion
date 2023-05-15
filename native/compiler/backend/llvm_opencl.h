#pragma once

#include "llvm.h"

namespace polyregion::backend {

class OpenCLLLVMTargetHandler : public LLVMTargetSpecificHandler {
  void witnessEntry(llvm::LLVMContext &ctx, llvm::Module &mod, llvm::Function &fn) override;
};

} // namespace polyregion::backend