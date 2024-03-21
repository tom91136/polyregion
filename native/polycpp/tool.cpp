#include <clang/CodeGen/BackendUtil.h>
#include <cstdlib>
#include <iostream>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Analysis/CallGraph.h"

#include "aspartame/string.hpp"
#include "aspartame/unordered_set.hpp"
#include "aspartame/variant.hpp"
#include "aspartame/vector.hpp"
#include "aspartame/view.hpp"

#include "frontend.h"
#include "options.h"
#include "rewriter.h"
#include "utils.hpp"

#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include <cxxabi.h>

using namespace aspartame;

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
  if (visited.find(F) != visited.end())
    return;
  visited.insert(F);
  for(auto &[_, node ] : *CG[F]){
    if(auto fn = node->getFunction(); fn  ){
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


  auto dnr = M //
             | collect([](const llvm::Function& F) {
                 return F.getName().str() ^ starts_with("__polyregion") //
                                || demangleCXXName(F.getName().data()) ^ starts_with("__polyregion")
                            ? std::optional{ &F}
                            : std::nullopt;
               }) //
             | map([&](const llvm::Function* F){
                 std::unordered_set<const llvm::Function*> tree;
                 traverseCallGraph(F, CG, tree);
                  return std::tuple{F, tree};
               })
             | to_vector();





    llvm::errs() << ">>>>\n" << (dnr ^ mk_string("\n", [](auto F, auto xs){
              return F->getName().str() + " = " + (xs^ mk_string(", ", [](auto ff) { return ff->getName().str() ;}));
                          }));

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

static std::vector<std::string> mkDelimitedEnvPaths(const char *env, std::optional<std::string> leading) {
  std::vector<std::string> xs;
  if (auto line = std::getenv(env); line) {
    for (auto &path : line ^ split(llvm::sys::EnvPathSeparator)) {
      if (leading) xs.push_back(*leading);
      xs.push_back(path);
    }
  }
  return xs;
};

int executeCC1(std::vector<std::string> &cc1Args, const std::optional<polyregion::polystl::StdParOptions> &stdpar) {



  if (stdpar) {

    auto includes = mkDelimitedEnvPaths("POLYSTL_INCLUDE", "-isystem");
    cc1Args.insert(cc1Args.end(), includes.begin(), includes.end());

    cc1Args.insert(cc1Args.end(), {"-include", "polystl/polystl.h"});

    // -static-polyrt
    // -dynamic-polyrt
    // -no-polyrt
  }

  std::vector<const char *> cc1Args_ = cc1Args ^ map([](auto &x) { return x.c_str(); });

  std::cout << (cc1Args ^ mk_string(" ")) << "\n";

  //  std::vector<const char *> cc1Args_(cc1Args.size());
  //  std::transform(cc1Args.begin(), cc1Args.end(), cc1Args_.begin(), [](auto &x) { return x.c_str(); });
  auto diagOptions = clang::CreateAndPopulateDiagOpts(cc1Args_);
  auto diagClient = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), &*diagOptions);
  auto diag = clang::CompilerInstance::createDiagnostics(diagOptions.release(), diagClient.release(), true);

  clang::CompilerInstance CI;
  CI.setDiagnostics(diag.get());
  if (!clang::CompilerInvocation::CreateFromArgs(CI.getInvocation(), cc1Args_, *diag)) {
    return EXIT_FAILURE;
  }

  auto ct = clang::TargetInfo::CreateTargetInfo(CI.getDiagnostics(), CI.getInvocation().TargetOpts);
  CI.setTarget(ct);

  if (CI.getFrontendOpts().ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return true;
  }

  bool success;
  if (stdpar && !std::getenv("POLYCPP_NO_REWRITE")) {
    using namespace polyregion;
    polystl::ModifyASTAndEmitObjAction action([&]() { return std::make_unique<polystl::OffloadRewriteConsumer>(CI.getDiagnostics(), *stdpar); });
    success = CI.ExecuteAction(action);
    if (!success) {
      diag->Report(diag->getCustomDiagID(clang::DiagnosticsEngine::Error, "[PolySTL] Frontend pass did not succeed"));
      return EXIT_FAILURE;
    }

    auto M = action.takeModule();

    if (stdpar->interposeMalloc) {
      interpose(*M);
    }

    clang::EmitBackendOutput(CI.getDiagnostics(),                           //
                             CI.getHeaderSearchOpts(),                      //
                             CI.getCodeGenOpts(),                           //
                             CI.getTargetOpts(),                            //
                             CI.getLangOpts(),                              //
                             M->getDataLayoutStr(),                         //
                             M.get(),                                       //
                             clang::Backend_EmitObj,                        //
                             CI.getFileManager().getVirtualFileSystemPtr(), //
                             CI.createDefaultOutputFile(true, "", "o"));

    CI.clearOutputFiles(/*erase*/ false);
    CI.getSourceManager().PrintStats();
  } else {


    clang::EmitObjAction action;
    success = CI.ExecuteAction(action);
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, const char *argv[]) {

  llvm::InitLLVM init(argc, argv);
  llvm::setBugReportMsg("PLEASE submit a bug report to TODO and include the crash backtrace, "
                        "preprocessed source, and associated run script.\n");

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  auto triple = llvm::sys::getDefaultTargetTriple();
  auto diagOptions = new clang::DiagnosticOptions();
  auto diagClient = new clang::TextDiagnosticPrinter(llvm::errs(), &*diagOptions);
  auto diagID = new clang::DiagnosticIDs();
  auto diags = clang::DiagnosticsEngine(diagID, diagOptions, diagClient);

  auto execPath = llvm::sys::fs::getMainExecutable(argv[0], nullptr);
  auto execParentDir = llvm::sys::path::parent_path(execPath).str();

  clang::driver::Driver driver(execPath, triple, diags, "PolyCpp compiler");
  driver.ResourceDir = (execParentDir + "/lib/clang/" + std::to_string(CLANG_VERSION_MAJOR));
  return polyregion::polystl::StdParOptions::stripAndParse({argv, argv + argc}) ^
         fold_total(
             [&](const std::vector<std::string> &errors) {
               driver.getDiags().Report(
                   diags.getCustomDiagID(clang::DiagnosticsEngine::Fatal, "Errors (%0) while parsing the following arguments:\n%1"))
                   << errors.size() << (errors ^ mk_string("\n") ^ indent(2));
               return EXIT_FAILURE;
             },
             [&](const std::pair<std::vector<const char *>, std::optional<polyregion::polystl::StdParOptions>> &v) {
               auto [args, stdpar] = v;

               // since the executable name won't be clang++ anymore, we manually set the mode to C++ by inserting the override
               // after the executable name
               args.insert(std::next(args.begin()), "--driver-mode=g++");

               if (!stdpar->quiet) {
                 driver.getDiags()
                     .Report(diags.getCustomDiagID(clang::DiagnosticsEngine::Remark, "[PolySTL] Targeting architectures: %0"))
                     .AddString(stdpar->targets ^
                                mk_string(", ", [](auto target, auto feature) { return std::string(to_string(target)) + "@" + feature; }));
               }

               //               if (stdpar) {
               auto libs = mkDelimitedEnvPaths("POLYSTL_LIB", {});
               std::transform(libs.begin(), libs.end(), std::back_inserter(args), [](auto &x) { return x.c_str(); });
               //               }

               std::unique_ptr<clang::driver::Compilation> compilation(driver.BuildCompilation(llvm::ArrayRef(args)));

               int returnCode = EXIT_SUCCESS;
               for (const auto &command : compilation->getJobs()) {
                 const auto &cmdArgs = command.getArguments();
                 if (command.getExecutable() == execPath &&                    // make sure the driver is actually calling us
                     command.getCreator().getName() == std::string("clang") && // and that clang is the compiler
                     cmdArgs[0] == std::string("-cc1")                         // and we're invoking the cc1 frontend
                 ) {
                   std::vector<std::string> actual;
                   actual.insert(actual.begin(), std::next(cmdArgs.begin()), cmdArgs.end()); //  skip the first -cc1
                   returnCode = executeCC1(actual, stdpar);
                   if (returnCode != EXIT_SUCCESS) break;
                 } else {
                   const clang::driver::Command *failed{};
                   if (auto code = compilation->ExecuteCommand(command, failed); code != EXIT_SUCCESS) {
                     driver.generateCompilationDiagnostics(*compilation, *failed);
                     returnCode = code;
                   }
                 }
               }
               diags.getClient()->finish();
               return returnCode;
             });
}