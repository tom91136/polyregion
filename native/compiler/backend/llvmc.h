#pragma once

#include "ast.h"
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

EXPORT polyast::Pair<polyast::Opt<std::string>, std::string> verifyModule(llvm::Module &mod);

EXPORT compiler::Compilation compileModule(const TargetInfo &info, const compiler::Opt &opt, bool emitDisassembly,
                                           std::unique_ptr<llvm::Module> M);

} // namespace polyregion::backend::llvmc
