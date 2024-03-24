#include "aspartame/string.hpp"
#include "aspartame/unordered_set.hpp"
#include "aspartame/vector.hpp"
#include "aspartame/view.hpp"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cxxabi.h>
#include <llvm/Analysis/CallGraph.h>
#include <unordered_set>

using namespace llvm;

namespace {

static std::string demangleCXXName(const char *abiName) {
  int failed;
  char *ret = abi::__cxa_demangle(abiName, nullptr /* output buffer */, nullptr /* length */, &failed);
  if (failed) {
    // 0: The demangling operation succeeded.
    // -1: A memory allocation failure occurred.
    // -2: mangled_name is not a valid name under the C++ ABI mangling rules.
    // -3: One of the arguments is invalid.
    return "";
  } else {
    return ret;
  }
}

void traverseCallGraph(const llvm::Function *F, llvm::CallGraph &CG, std::unordered_set<const llvm::Function *> &visited) {
  if (visited.find(F) != visited.end()) return;
  visited.insert(F);
  for (auto &[_, node] : *CG[F]) {
    if (auto fn = node->getFunction(); fn) {
      traverseCallGraph(fn, CG, visited);
    }
  }
}

void interpose(llvm::Module &M) {
  using namespace llvm;

  static constexpr std::pair<StringLiteral, StringLiteral> ReplaceMap[]{
      //      {"aligned_alloc", "__polyregion_aligned_alloc"},
      //      {"calloc", "__polyregion_calloc"},
      //      {"free", "__polyregion_free"},
      //      {"malloc", "__polyregion_malloc"},
      //      {"memalign", "__polyregion_aligned_alloc"},
      //      {"posix_memalign", "__polyregion_posix_aligned_alloc"},
      //      {"realloc", "__polyregion_realloc"},
      //      {"reallocarray", "__polyregion_realloc_array"},
      {"_ZdaPv", "__polyregion_operator_delete"},
      //      {"_ZdaPvm", "__polyregion_operator_delete_sized"},
      //      {"_ZdaPvSt11align_val_t", "__polyregion_operator_delete_aligned"},
      //      {"_ZdaPvmSt11align_val_t", "__polyregion_operator_delete_aligned_sized"},
      {"_ZdlPv", "__polyregion_operator_delete"},
      //      {"_ZdlPvm", "__polyregion_operator_delete_sized"},
      //      {"_ZdlPvSt11align_val_t", "__polyregion_operator_delete_aligned"},
      //      {"_ZdlPvmSt11align_val_t", "__polyregion_operator_delete_aligned_sized"},
      {"_Znam", "__polyregion_operator_new"},
      //      {"_ZnamRKSt9nothrow_t", "__polyregion_operator_new_nothrow"},
      //      {"_ZnamSt11align_val_t", "__polyregion_operator_new_aligned"},
      //      {"_ZnamSt11align_val_tRKSt9nothrow_t", "__polyregion_operator_new_aligned_nothrow"},

      {"_Znwm", "__polyregion_operator_new"},
      //      {"_ZnwmRKSt9nothrow_t", "__polyregion_operator_new_nothrow"},
      //      {"_ZnwmSt11align_val_t", "__polyregion_operator_new_aligned"},
      //      {"_ZnwmSt11align_val_tRKSt9nothrow_t", "__polyregion_operator_new_aligned_nothrow"},
      //      {"__builtin_calloc", "__polyregion_calloc"},
      //      {"__builtin_free", "__polyregion_free"},
      //      {"__builtin_malloc", "__polyregion_malloc"},
      //      {"__builtin_operator_delete", "__polyregion_operator_delete"},
      //      {"__builtin_operator_new", "__polyregion_operator_new"},
      //      {"__builtin_realloc", "__polyregion_realloc"},
      //      {"__libc_calloc", "__polyregion_calloc"},
      //      {"__libc_free", "__polyregion_free"},
      //      {"__libc_malloc", "__polyregion_malloc"},
      //      {"__libc_memalign", "__polyregion_aligned_alloc"},
      //      {"__libc_realloc", "__polyregion_realloc"}
  };

  SmallDenseMap<StringRef, StringRef> AllocReplacements(std::cbegin(ReplaceMap), std::cend(ReplaceMap));

  llvm::CallGraph CG(M);

  using namespace aspartame;

  auto dnr = M //
             | collect([](const llvm::Function &F) {
                 return F.getName().str() ^ starts_with("__polyregion")
                            ? std::optional{&F}
                            : std::nullopt;
               }) //
             | map([&](const llvm::Function *F) {
                 std::unordered_set<const llvm::Function *> tree;
                 traverseCallGraph(F, CG, tree);
                 return std::tuple{F, tree};
               }) |
             to_vector();

  llvm::errs() << ">>>>\n"
               << (dnr ^ mk_string("\n", [](auto F, auto xs) {
                     return F->getName().str() + " = " + (xs ^ filter([&](auto x ) { return x != F; }) ^ mk_string(", ", [](auto ff) { return ff->getName().str(); }));
                   }));
  llvm::errs() << "===\n";

  for (auto &&F : M) {

    llvm::errs() << "Fn: " << F.getName() << " cxx=" << demangleCXXName(F.getName().data()) << "\n";

    for (auto &i : F) {
      for (auto &instr : i) {

        // sealed: local alloca
        // unsealed:
        //   -c(unknown)           => cross TU, NEEDS LINKER, split comp
        //   - p -> c(stack_ptr)   => pass information
        //   - p -> c?(stack_ptr)  => special case c
        //   - p -> c(stack_ptr?)  => special ptr

        //
        //
        //
        //        if (llvm::isa<llvm::AllocaInst>(instr)) {
        //
        //
        //          llvm::errs() << "Fn " << F.getName() << " " ;
        //
        //          llvm::AllocaInst &allocaInst = llvm::cast<llvm::AllocaInst>(instr);
        //
        //          // Accessing the allocated type
        //          llvm::Type *allocatedType = allocaInst.getAllocatedType();
        //          llvm::errs() << "Allocated Type: ";
        //          allocatedType->print(llvm::errs());
        //          llvm::errs() << "\n";
        //
        //          if (allocaInst.isArrayAllocation()) {
        //            llvm::Value *arraySize = allocaInst.getArraySize();
        //            llvm::errs() << "Array size (as an operand): ";
        //            arraySize->print(llvm::errs());
        //            llvm::errs() << "\n";
        //          }
        //        }
      }
    }
    if (!F.hasName()) continue;
    if (!AllocReplacements.contains(F.getName())) continue;

    if (auto R = M.getFunction(AllocReplacements[F.getName()])) {
      //      F.replaceAllUsesWith(R);
      F.replaceUsesWithIf(R, [](llvm::Use &u) {
        //        llvm::errs() << ">>> " << u. << "\n";

        return true;
      });

    } else {
      std::string W;
      raw_string_ostream OS(W);

      OS << "cannot be interposed, missing: " << AllocReplacements[F.getName()]
         << ". Tried to run the allocation interposition pass without the "
         << "replacement functions available.";

      F.getContext().diagnose(DiagnosticInfoUnsupported(F, W, F.getSubprogram(), DS_Warning));
    }
  }
}

struct PolySTLInterposePass : public PassInfoMixin<PolySTLInterposePass> {

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) { // NOLINT(*-convert-member-functions-to-static)
    interpose(M);
    return PreservedAnalyses::none();
  }
};

} // end anonymous namespace

llvm::PassPluginLibraryInfo getInterposePluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "PolySTLInterposePass", LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerOptimizerLastEPCallback(
                [](llvm::ModulePassManager &PM, OptimizationLevel Level) { PM.addPass(PolySTLInterposePass()); });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() { return getInterposePluginInfo(); }
