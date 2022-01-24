#pragma once

#include "compiler.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

namespace polyregion::backend::llvmc {

void initialise();

const llvm::TargetMachine &targetMachine();

polyregion::compiler::Compilation compileModule(bool emitDisassembly,            //
                                                std::unique_ptr<llvm::Module> M, //
                                                llvm::LLVMContext &Context       //
);

} // namespace polyregion::backend::llvmc
