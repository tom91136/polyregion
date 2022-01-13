#pragma once

#include "compiler.h"
#include "llvm/IR/Module.h"

namespace polyregion::backend::llvmc {

void initialise();

polyregion::compiler::Compilation compileModule(bool emitDisassembly,            //
                                                std::unique_ptr<llvm::Module> M, //
                                                llvm::LLVMContext &Context       //
);

} // namespace polyregion::backend::llvmc
