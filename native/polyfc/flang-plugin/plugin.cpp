// clang-format off
// XXX Expr<T> must be complete before <vector>/<memory> is included transitively; clang-cl
// otherwise eagerly instantiates std::vector<Expr<T>>'s dtor type-traits and trips on the
// forward declaration. Regroup would otherwise sort this after <vector>.
#include "flang/Evaluate/expression.h"
// clang-format on

#include <fstream>
#include <vector>

#if defined(__APPLE__)
  #include <crt_externs.h>
#elif defined(_WIN32)
// XXX Forward-declare the Win32 command-line APIs; including <windows.h> clashes with flang's
// macro vocabulary.
extern "C" {
__declspec(dllimport) wchar_t *__stdcall GetCommandLineW(void);
__declspec(dllimport) wchar_t **__stdcall CommandLineToArgvW(const wchar_t *lpCmdLine, int *pNumArgs);
__declspec(dllimport) void *__stdcall LocalFree(void *hMem);
__declspec(dllimport) int __stdcall WideCharToMultiByte(unsigned CodePage, unsigned long dwFlags, const wchar_t *lpWideCharStr,
                                                        int cchWideChar, char *lpMultiByteStr, int cbMultiByte, const char *lpDefaultChar,
                                                        int *lpUsedDefaultChar);
}
#endif

#include "clang/Options/Options.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Option/ArgList.h"

#include "rewriter.h"

namespace {

std::vector<std::string> getCmdLine() {
#ifdef __linux__
  std::ifstream cmdline("/proc/self/cmdline");
  if (!cmdline) {
    std::fprintf(stderr, "Failed to open /proc/self/cmdline\n");
    std::abort();
  }
  std::vector<std::string> args;
  std::string line;
  while (std::getline(cmdline, line, '\0'))
    args.push_back(line);
  return args;
#elif defined(__APPLE__)
  char ***argvp = _NSGetArgv();
  int *argcp = _NSGetArgc();

  std::vector<std::string> args;
  for (int i = 0; i < *argcp; ++i) {
    args.push_back(argvp[0][i]);
  }
  return args;
#elif defined(_WIN32)
  int argc = 0;
  wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
  std::vector<std::string> args;
  if (!wargv) return args;
  for (int i = 0; i < argc; ++i) {
    const unsigned CP_UTF8 = 65001;
    int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
    std::string s(len > 0 ? len - 1 : 0, '\0');
    if (len > 0) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), len, nullptr, nullptr);
    args.push_back(std::move(s));
  }
  LocalFree(wargv);
  return args;
#else
  #error "getCmdLine unimplemented for OS"
#endif
}

class MutableCodeGenAction final : public Fortran::frontend::CodeGenAction {
public:
  explicit MutableCodeGenAction(const Fortran::frontend::BackendActionTy ty) : CodeGenAction(ty) {}
  void setAction(const Fortran::frontend::BackendActionTy ty) { action = ty; }
  mlir::ModuleOp getMLIRModule() const {
#if LLVM_VERSION_MAJOR >= 20
    return mlirModule.get();
#else
    return *mlirModule.get();
#endif
  }

  template <typename F>
  static bool executeInterposedMLIRAction(FrontendAction &parent, const Fortran::frontend::BackendActionTy ty, //
                                          F witnessHLFIR, F witnessFIR) {

    auto &ci = parent.getInstance();
    if (ty == Fortran::frontend::BackendActionTy::Backend_EmitHLFIR) {
      ci.getDiagnostics().Report(ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Note, "Rewriting FIR but HLFIR output requested; HLFIR will not include tree rewrites"));
      MutableCodeGenAction action(ty);
      return ci.executeAction(action);
    } else {
      MutableCodeGenAction action(Fortran::frontend::BackendActionTy::Backend_EmitHLFIR);
      ci.getInvocation().getLoweringOpts().setLowerToHighLevelFIR(true);
      ci.executeAction(action);

      if (auto m = action.getMLIRModule()) witnessHLFIR(ci.getDiagnostics(), m);
      else {
        llvm::errs() << "Lower to HLFIR resulted null MLIR module\n";
        std::abort();
      }

      action.setInstance(&ci);
      action.setCurrentInput(parent.getCurrentInput());
      action.lowerHLFIRToFIR();
      if (auto m = action.getMLIRModule()) witnessFIR(ci.getDiagnostics(), m);
      else {
        llvm::errs() << "Lower to FIR resulted null MLIR module\n";
        std::abort();
      }

      action.setAction(ty);
      action.setInstance(&ci);
      action.setCurrentInput(parent.getCurrentInput());

      // XXX .init_array should be used otherwise llvm.global_ctor don't work
      //  In Clang, this is the default unless OPT_fno_use_init_array is specified.
      //  Fortran doesn't have support complex ctor on module-load, so nothing lowers to the llvm.global_ctor
      parent.getInstance().getTargetMachine().Options.UseInitArray = true;

      if (llvm::Error err = action.execute()) {
        llvm::consumeError(std::move(err));
        return false;
      }
      return true;
    }
  }
};

template <typename A> static bool executeAction(Fortran::frontend::FrontendAction &parent) {
  A action;
  return parent.getInstance().executeAction(action);
}

class RewriteIRAction final : public Fortran::frontend::PluginParseTreeAction {

  void executeAction() override {
    Fortran::semantics::SemanticsContext &ctx = getInstance().getSemanticsContext();
    llvm::outs() << "[PolyFC] opts=" << ctx.targetCharacteristics().compilerOptionsString() << "\n";

    const auto cmdArgs = getCmdLine();
    std::vector<const char *> cmdArgRef;
    for (auto &x : cmdArgs)
      cmdArgRef.push_back(x.c_str());
    unsigned int missingArgIndex{};
    unsigned int missingArgCount{};
    const auto args =
        clang::getDriverOptTable().ParseArgs(cmdArgRef, missingArgIndex, missingArgCount, llvm::opt::Visibility(clang::options::FC1Option));
    if (const auto arg = args.getLastArg(clang::options::OPT_Action_Group)) {
      llvm::outs() << "[PolyFC] action=" << arg->getOption().getName() << "\n";
      const bool success = [&] {
        using namespace Fortran::frontend;
        using namespace clang::options;
        const auto interposeAction = [&](const BackendActionTy ty) {
          return MutableCodeGenAction::executeInterposedMLIRAction(*this, ty,                         //
                                                                   &polyregion::polyfc::rewriteHLFIR, //
                                                                   &polyregion::polyfc::rewriteFIR);
        };
        switch (arg->getOption().getID()) {
          // built-in
          case OPT_test_io: return ::executeAction<InputOutputTestAction>(*this);
          case OPT_E: return ::executeAction<PrintPreprocessedAction>(*this);

          case OPT_fsyntax_only: return ::executeAction<ParseSyntaxOnlyAction>(*this);
          case OPT_fdebug_unparse: return ::executeAction<DebugUnparseAction>(*this);
          case OPT_fdebug_unparse_no_sema: return ::executeAction<DebugUnparseNoSemaAction>(*this);
          case OPT_fdebug_unparse_with_symbols: return ::executeAction<DebugUnparseWithSymbolsAction>(*this);
          case OPT_fdebug_unparse_with_modules: return ::executeAction<DebugUnparseWithModulesAction>(*this);
          case OPT_fdebug_dump_symbols: return ::executeAction<DebugDumpSymbolsAction>(*this);
          case OPT_fdebug_dump_parse_tree: return ::executeAction<DebugDumpParseTreeAction>(*this);
          case OPT_fdebug_dump_pft: return ::executeAction<DebugDumpPFTAction>(*this);
          case OPT_fdebug_dump_all: return ::executeAction<DebugDumpParseTreeNoSemaAction>(*this);
          case OPT_fdebug_dump_parse_tree_no_sema: return ::executeAction<DebugDumpAllAction>(*this);
          case OPT_fdebug_dump_provenance: return ::executeAction<DebugDumpProvenanceAction>(*this);
          case OPT_fdebug_dump_parsing_log: return ::executeAction<DebugDumpParsingLogAction>(*this);
          case OPT_fdebug_measure_parse_tree: return ::executeAction<DebugMeasureParseTreeAction>(*this);
          case OPT_fdebug_pre_fir_tree: return ::executeAction<DebugPreFIRTreeAction>(*this);
          case OPT_fget_symbols_sources: return ::executeAction<GetDefinitionAction>(*this);
          case OPT_fget_definition: return ::executeAction<GetSymbolsSourcesAction>(*this);
          case OPT_init_only: return ::executeAction<InitOnlyAction>(*this);
          // interposed ones
          case OPT_emit_hlfir: return interposeAction(BackendActionTy::Backend_EmitHLFIR);
          case OPT_emit_fir: return interposeAction(BackendActionTy::Backend_EmitFIR);
          case OPT_emit_llvm: return interposeAction(BackendActionTy::Backend_EmitLL);
          case OPT_emit_llvm_bc: return interposeAction(BackendActionTy::Backend_EmitBC);
          case OPT_emit_obj: return interposeAction(BackendActionTy::Backend_EmitObj);
          case OPT_S: return interposeAction(BackendActionTy::Backend_EmitAssembly);
          default:
            getInstance().getDiagnostics().Report(
                getInstance().getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error, "Unhandled plugin action option: %0"))
                << arg->getOption().getName();
            return false;
        }
      }();
      if (!success) {
        getInstance().getDiagnostics().Report(
            getInstance().getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error, "Plugin redirected action failed"));
      }
    }
  }
};
} // namespace

[[maybe_unused]] static Fortran::frontend::FrontendPluginRegistry::Add<RewriteIRAction> X("polyfc", "");
