#include "llvm_nvptx.h"

void polyregion::backend::NVPTXLLVMTargetHandler::witnessEntry(llvm::LLVMContext &ctx, llvm::Module &mod, llvm::Function &fn) {
  mod.getOrInsertNamedMetadata("nvvm.annotations")
      ->addOperand(llvm::MDNode::get(ctx, // XXX the attribute name must be "kernel" here and not the function name!
                                     {llvm::ValueAsMetadata::get(&fn), llvm::MDString::get(ctx, "kernel"),
                                      llvm::ValueAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))}));
}
