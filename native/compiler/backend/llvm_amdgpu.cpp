#include "llvm_amdgpu.h"

void polyregion::backend::AMDGPULLVMTargetHandler::witnessEntry(llvm::LLVMContext &ctx, llvm::Module &mod, llvm::Function &fn) {
  fn.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
}
