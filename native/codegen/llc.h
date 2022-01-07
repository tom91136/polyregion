#pragma once

#include "llvm/IR/Module.h"
namespace llc {

void setup();
int compileModule(std::unique_ptr<llvm::Module> M, llvm::LLVMContext &Context);

} // namespace llc
