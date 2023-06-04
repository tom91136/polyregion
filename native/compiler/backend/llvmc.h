#pragma once

#include "ast.h"
#include "compiler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace polyregion::backend::llvmc {

POLYREGION_EXPORT void initialise();

POLYREGION_EXPORT llvm::Triple defaultHostTriple();

struct POLYREGION_EXPORT CpuInfo {
  POLYREGION_EXPORT std::string uArch, features;
};

struct POLYREGION_EXPORT TargetInfo {
  POLYREGION_EXPORT llvm::Triple triple;
  POLYREGION_EXPORT std::optional<llvm::DataLayout> layout;
  POLYREGION_EXPORT const llvm::Target *target;
  POLYREGION_EXPORT CpuInfo cpu;
  POLYREGION_EXPORT llvm::DataLayout resolveDataLayout() const;
};

POLYREGION_EXPORT const CpuInfo &hostCpuInfo();
POLYREGION_EXPORT const llvm::Target *targetFromTriple(const llvm::Triple &tripleName);
POLYREGION_EXPORT polyast::Pair<polyast::Opt<std::string>, std::string> verifyModule(llvm::Module &mod);
POLYREGION_EXPORT compiler::Compilation compileModule(const TargetInfo &info, const compiler::Opt &opt, bool emitDisassembly,
                                                      std::unique_ptr<llvm::Module> M);

} // namespace polyregion::backend::llvmc
