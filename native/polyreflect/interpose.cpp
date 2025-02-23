#include <cxxabi.h>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"

#include "aspartame/all.hpp"

#include "interpose.h"

using namespace aspartame;
using namespace polyregion;

namespace {

std::string demangleCXXName(const char *abiName) {
  int failed;
  char *ret = abi::__cxa_demangle(abiName, nullptr /* output buffer */, nullptr /* length */, &failed);
  if (failed) {
    //  0: The demangling operation succeeded.
    // -1: A memory allocation failure occurred.
    // -2: mangled_name is not a valid name under the C++ ABI mangling rules.
    // -3: One of the arguments is invalid.
    return "";
  } else {
    return ret;
  }
}

bool runSplice(llvm::Module &M, const bool verbose) {
  static constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral> ReplaceMap[]{
      {"aligned_alloc", "polyrt_usm_aligned_alloc"},
      //      {"calloc", "polyrt_usm_calloc"},
      {"free", "polyrt_usm_free"},
      {"malloc", "polyrt_usm_malloc"},
      //      {"memalign", "polyrt_usm_aligned_alloc"},
      //      {"posix_memalign", "polyrt_usm_posix_aligned_alloc"},
      //      {"realloc", "polyrt_usm_realloc"},
      //      {"reallocarray", "polyrt_usm_realloc_array"},
      {"_ZdaPv", "polyrt_usm_operator_delete"},
      //      {"_ZdaPvm", "polyrt_usm_operator_delete_sized"},
      //      {"_ZdaPvSt11align_val_t", "polyrt_usm_operator_delete_aligned"},
      //      {"_ZdaPvmSt11align_val_t", "polyrt_usm_operator_delete_aligned_sized"},
      {"_ZdlPv", "polyrt_usm_operator_delete"},
      {"_ZdlPvm", "polyrt_usm_operator_delete_sized"},
      //      {"_ZdlPvSt11align_val_t", "polyrt_usm_operator_delete_aligned"},
      //      {"_ZdlPvmSt11align_val_t", "polyrt_usm_operator_delete_aligned_sized"},
      {"_Znam", "polyrt_usm_operator_new"},
      //      {"_ZnamRKSt9nothrow_t", "polyrt_usm_operator_new_nothrow"},
      //      {"_ZnamSt11align_val_t", "polyrt_usm_operator_new_aligned"},
      //      {"_ZnamSt11align_val_tRKSt9nothrow_t", "polyrt_usm_operator_new_aligned_nothrow"},

      {"_Znwm", "polyrt_usm_operator_new"},
      //      {"_ZnwmRKSt9nothrow_t", "polyrt_usm_operator_new_nothrow"},
      //      {"_ZnwmSt11align_val_t", "polyrt_usm_operator_new_aligned"},
      //      {"_ZnwmSt11align_val_tRKSt9nothrow_t", "polyrt_usm_operator_new_aligned_nothrow"},
      //      {"__builtin_calloc", "polyrt_usm_calloc"},
      {"__builtin_free", "polyrt_usm_free"},
      {"__builtin_malloc", "polyrt_usm_malloc"},
      //      {"__builtin_operator_delete", "polyrt_usm_operator_delete"},
      //      {"__builtin_operator_new", "polyrt_usm_operator_new"},
      //      {"__builtin_realloc", "polyrt_usm_realloc"},
      //      {"__libc_calloc", "polyrt_usm_calloc"},
      {"__libc_free", "polyrt_usm_free"},
      {"__libc_malloc", "polyrt_usm_malloc"},
      //      {"__libc_memalign", "polyrt_usm_aligned_alloc"},
      //      {"__libc_realloc", "polyrt_usm_realloc"}
  };
  llvm::SmallDenseMap<llvm::StringRef, llvm::StringRef> AllocReplacements(std::cbegin(ReplaceMap), std::cend(ReplaceMap));
  bool modified = false;
  for (auto &F : M) {
    if (!F.hasName()) continue;
    if (!AllocReplacements.contains(F.getName())) continue;

    if (verbose) llvm::errs() << "[InterposePass] In " << F.getName() << " (demangled=" << demangleCXXName(F.getName().data()) << ")\n";
    if (const auto Replacement = M.getFunction(AllocReplacements[F.getName()])) {
      F.replaceUsesWithIf(Replacement, [](llvm::Use &u) {
        llvm::errs() << "[InterposePass]   Interposed " << u << "\n";
        return true;
      });
      modified = true;
    } else {
      std::string W;
      llvm::raw_string_ostream OS(W);
      OS << F.getName() << " cannot be interposed, missing: " << AllocReplacements[F.getName()]
         << ". Tried to run the allocation interposition pass without the "
         << "replacement function in module.";
      F.getContext().diagnose(llvm::DiagnosticInfoUnsupported(F, W, F.getSubprogram(), llvm::DS_Warning));
    }
  }
  return modified;
}

} // namespace

polyreflect::InterposePass::InterposePass(const bool verbose) : verbose(verbose) {}
llvm::PreservedAnalyses polyreflect::InterposePass::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  if (runSplice(M, verbose)) return llvm::PreservedAnalyses::none();
  return llvm::PreservedAnalyses::all();
}