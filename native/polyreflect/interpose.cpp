#include "interpose.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"

#include "aspartame/all.hpp"

#include "polyregion/msvc_abi_names.h"

using namespace aspartame;
using namespace polyregion;

namespace {

constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral> ReplaceMap[]{
    {"aligned_alloc", "aligned_alloc"},
    //      {"calloc", "calloc"},
    {"free", "free"},
    {"malloc", "malloc"},
    //      {"memalign", "aligned_alloc"},
    //      {"posix_memalign", "posix_aligned_alloc"},
    //      {"realloc", "realloc"},
    //      {"reallocarray", "realloc_array"},
    // Itanium C++ ABI new/delete:
    {"_ZdaPv", "operator_delete"},
    //      {"_ZdaPvm", "operator_delete_sized"},
    //      {"_ZdaPvSt11align_val_t", "operator_delete_aligned"},
    //      {"_ZdaPvmSt11align_val_t", "operator_delete_aligned_sized"},
    {"_ZdlPv", "operator_delete"},
    {"_ZdlPvm", "operator_delete_sized"},
    //      {"_ZdlPvSt11align_val_t", "operator_delete_aligned"},
    //      {"_ZdlPvmSt11align_val_t", "operator_delete_aligned_sized"},
    {"_Znam", "operator_new"},
    //      {"_ZnamRKSt9nothrow_t", "operator_new_nothrow"},
    //      {"_ZnamSt11align_val_t", "operator_new_aligned"},
    //      {"_ZnamSt11align_val_tRKSt9nothrow_t", "operator_new_aligned_nothrow"},

    {"_Znwm", "operator_new"},
    //      {"_ZnwmRKSt9nothrow_t", "operator_new_nothrow"},
    //      {"_ZnwmSt11align_val_t", "operator_new_aligned"},
    //      {"_ZnwmSt11align_val_tRKSt9nothrow_t", "operator_new_aligned_nothrow"},
    //      {"__builtin_calloc", "calloc"},
    {"__builtin_free", "free"},
    {"__builtin_malloc", "malloc"},
    //      {"__builtin_operator_delete", "operator_delete"},
    //      {"__builtin_operator_new", "operator_new"},
    //      {"__builtin_realloc", "realloc"},
    //      {"__libc_calloc", "calloc"},
    {"__libc_free", "free"},
    {"__libc_malloc", "malloc"},
    //      {"__libc_memalign", "aligned_alloc"},
    //      {"__libc_realloc", "realloc"}
    //
    // MSVC x64 C++ ABI new/delete. See https://learn.microsoft.com/cpp/build/reference/decorated-names
    {msvc_abi::OperatorNew, "operator_new"},                       // operator new(size_t)
    {msvc_abi::OperatorNewArray, "operator_new"},                  // operator new[](size_t)
    {msvc_abi::OperatorDelete, "operator_delete"},                 // operator delete(void*)
    {msvc_abi::OperatorDeleteArray, "operator_delete"},            // operator delete[](void*)
    {msvc_abi::OperatorDeleteSized, "operator_delete_sized"},      // operator delete(void*, size_t)
    {msvc_abi::OperatorDeleteArraySized, "operator_delete_sized"}, // operator delete[](void*, size_t)
};

bool runSplice(llvm::Module &M, const llvm::StringRef tag, const llvm::StringRef targetPrefix, const bool verbose) {
  llvm::SmallDenseMap<llvm::StringRef, llvm::StringRef> AllocReplacements(std::cbegin(ReplaceMap), std::cend(ReplaceMap));
  bool modified = false;
  for (auto &F : M) {
    if (!F.hasName()) continue;
    if (!AllocReplacements.contains(F.getName())) continue;

    const auto replacement = (targetPrefix + AllocReplacements[F.getName()]).str();
    if (verbose)
      llvm::errs() << "[" << tag << "] In " << F.getName() << " (demangled=" << llvm::demangle(F.getName()) << ") -> " << replacement
                   << "\n";
    const auto Replacement = M.getOrInsertFunction(replacement, F.getFunctionType()).getCallee();
    if (verbose) {
      for (auto &u : F.uses())
        llvm::errs() << "[" << tag << "]   Interposed " << u << "\n";
    }
    F.replaceAllUsesWith(Replacement);
    modified = true;
  }
  return modified;
}

} // namespace

polyreflect::InterposePass::InterposePass(const bool verbose) : verbose(verbose) {}
llvm::PreservedAnalyses polyreflect::InterposePass::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  if (runSplice(M, "InterposePass", "polyrt_usm_", verbose)) return llvm::PreservedAnalyses::none();
  return llvm::PreservedAnalyses::all();
}

polyreflect::RecordAllocPass::RecordAllocPass(const bool verbose) : verbose(verbose) {}
llvm::PreservedAnalyses polyreflect::RecordAllocPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  if (runSplice(M, "RecordAllocPass", "polyrt_record_", verbose)) return llvm::PreservedAnalyses::none();
  return llvm::PreservedAnalyses::all();
}
