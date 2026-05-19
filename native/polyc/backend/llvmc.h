#pragma once

#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include "polyregion/compat.h"

#include "ast.h"
#include "compiler.h"

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
POLYREGION_EXPORT Pair<Opt<std::string>, std::string> verifyModule(llvm::Module &mod);
POLYREGION_EXPORT polyast::CompileResult compileModule(const TargetInfo &info, const compiletime::OptLevel &opt, bool emitDisassembly,
                                                       llvm::Module &M);

POLYREGION_EXPORT std::string findInDirs(llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> dirs);
POLYREGION_EXPORT std::string findVendorBitcode(llvm::StringRef name);
POLYREGION_EXPORT bool linkVendorBitcodeFile(llvm::Module &M, llvm::StringRef path);

} // namespace polyregion::backend::llvmc
