#pragma once

#include "compiler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace polyregion::backend::llvmc {

EXPORT void initialise();

EXPORT llvm::Triple defaultHostTriple();

struct EXPORT CpuInfo {
  EXPORT std::string uArch, features;
};

struct EXPORT TargetInfo {
  EXPORT llvm::Triple triple;
  EXPORT llvm::Target target;
  EXPORT CpuInfo cpu;
};

EXPORT const CpuInfo &hostCpuInfo();
EXPORT const llvm::Target &targetFromTriple(const llvm::Triple &tripleName);

EXPORT std::unique_ptr<llvm::TargetMachine> targetMachineFromTarget(const TargetInfo &info);

EXPORT compiler::Compilation compileModule(const TargetInfo &info, const compiler::Opt &opt, bool emitDisassembly,
                                           std::unique_ptr<llvm::Module> M, llvm::LLVMContext &Context);

} // namespace polyregion::backend::llvmc
