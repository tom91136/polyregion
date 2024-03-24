#include "driver_clang.h"

#include "clang/Basic/Stack.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Config/config.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AArch64TargetParser.h"

#ifdef CLANG_HAVE_RLIMITS
  #include <clang/Frontend/ChainedDiagnosticConsumer.h>
  #include <clang/Frontend/SerializedDiagnosticPrinter.h>
  #include <clang/StaticAnalyzer/Frontend/AnalyzerHelpFlags.h>
  #include <llvm/Support/InitLLVM.h>
  #include <sys/resource.h>
#endif

// This file is mostly migrated from https://github.com/llvm/llvm-project/blob/main/clang/tools/driver/cc1_main.cpp

using namespace clang;
using namespace llvm::opt;

static void LLVMErrorHandler(void *UserData, const char *Message, bool GenCrashDiag) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine *>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // Run the interrupt handlers to make sure any special cleanups get done, in
  // particular that we remove files registered with RemoveFileOnSignal.
  llvm::sys::RunInterruptHandlers();

  // We cannot recover from llvm errors.  When reporting a fatal error, exit
  // with status 70 to generate crash diagnostics.  For BSD systems this is
  // defined as an internal software error.  Otherwise, exit with status 1.
  llvm::sys::Process::Exit(GenCrashDiag ? 70 : 1);
}

#ifdef CLANG_HAVE_RLIMITS
/// Attempt to ensure that we have at least 8MiB of usable stack space.
static void ensureSufficientStack() {
  struct rlimit rlim;
  if (getrlimit(RLIMIT_STACK, &rlim) != 0) return;

  // Increase the soft stack limit to our desired level, if necessary and
  // possible.
  if (rlim.rlim_cur != RLIM_INFINITY && rlim.rlim_cur < rlim_t(DesiredStackSize)) {
    // Try to allocate sufficient stack.
    if (rlim.rlim_max == RLIM_INFINITY || rlim.rlim_max >= rlim_t(DesiredStackSize)) rlim.rlim_cur = DesiredStackSize;
    else if (rlim.rlim_cur == rlim.rlim_max) return;
    else rlim.rlim_cur = rlim.rlim_max;

    if (setrlimit(RLIMIT_STACK, &rlim) != 0 || rlim.rlim_cur != DesiredStackSize) return;
  }
}
#else
static void ensureSufficientStack() {}
#endif

/// Print supported cpus of the given target.
static int PrintSupportedCPUs(std::string TargetStr) {
  std::string Error;
  const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget(TargetStr, Error);
  if (!TheTarget) {
    llvm::errs() << Error;
    return 1;
  }

  // the target machine will handle the mcpu printing
  llvm::TargetOptions Options;
  std::unique_ptr<llvm::TargetMachine> TheTargetMachine(TheTarget->createTargetMachine(TargetStr, "", "+cpuhelp", Options, std::nullopt));
  return 0;
}

// TODO LLVM 18
// static int PrintSupportedExtensions(std::string TargetStr) {
//  std::string Error;
//  const llvm::Target *TheTarget =
//      llvm::TargetRegistry::lookupTarget(TargetStr, Error);
//  if (!TheTarget) {
//    llvm::errs() << Error;
//    return 1;
//  }
//
//  llvm::TargetOptions Options;
//  std::unique_ptr<llvm::TargetMachine> TheTargetMachine(
//      TheTarget->createTargetMachine(TargetStr, "", "", Options, std::nullopt));
//  const llvm::Triple &MachineTriple = TheTargetMachine->getTargetTriple();
//  const llvm::MCSubtargetInfo *MCInfo = TheTargetMachine->getMCSubtargetInfo();
//  const llvm::ArrayRef<llvm::SubtargetFeatureKV> Features =
//      MCInfo->getAllProcessorFeatures();
//
//  llvm::StringMap<llvm::StringRef> DescMap;
//  for (const llvm::SubtargetFeatureKV &feature : Features)
//    DescMap.insert({feature.Key, feature.Desc});
//
//  if (MachineTriple.isRISCV())
//    llvm::riscvExtensionsHelp(DescMap);
//  else if (MachineTriple.isAArch64())
//    llvm::AArch64::PrintSupportedExtensions(DescMap);
//  else if (MachineTriple.isARM())
//    llvm::ARM::PrintSupportedExtensions(DescMap);
//  else {
//    // The option was already checked in Driver::HandleImmediateArgs,
//    // so we do not expect to get here if we are not a supported architecture.
//    assert(0 && "Unhandled triple for --print-supported-extensions option.");
//    return 1;
//  }
//
//  return 0;
//}

int polyregion::polystl::useCI(llvm::ArrayRef<const char *> Argv, const char *Argv0, void *MainAddr,
                               const std::function<int(clang::CompilerInstance &)> &f) {

  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Register the support for object-file-wrapped Clang modules.
  auto PCHOps = Clang->getPCHContainerOperations();
  PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  // Setup round-trip remarks for the DiagnosticsEngine used in CreateFromArgs.
  if (find(Argv, StringRef("-Rround-trip-cc1-args")) != Argv.end())
    Diags.setSeverity(diag::remark_cc1_round_trip_generated, diag::Severity::Remark, {});

  bool Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(), Argv, Diags, Argv0);

  if (!Clang->getFrontendOpts().TimeTracePath.empty()) {
    llvm::timeTraceProfilerInitialize(Clang->getFrontendOpts().TimeTraceGranularity, Argv0);
  }
  // --print-supported-cpus takes priority over the actual compilation.
  if (Clang->getFrontendOpts().PrintSupportedCPUs) return PrintSupportedCPUs(Clang->getTargetOpts().Triple);

  // TODO LLVM 18
  //  // --print-supported-extensions takes priority over the actual compilation.
  //  if (Clang->getFrontendOpts().PrintSupportedExtensions)
  //    return PrintSupportedExtensions(Clang->getTargetOpts().Triple);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes && Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir = CompilerInvocation::GetResourcesPath(Argv0, MainAddr);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics();
  if (!Clang->hasDiagnostics()) return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(LLVMErrorHandler, static_cast<void *>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success) {
    Clang->getDiagnosticClient().finish();
    return 1;
  }

  int code = f(*Clang);

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());
  llvm::TimerGroup::clearAll();

  if (llvm::timeTraceProfilerEnabled()) {
    // It is possible that the compiler instance doesn't own a file manager here
    // if we're compiling a module unit. Since the file manager are owned by AST
    // when we're compiling a module unit. So the file manager may be invalid
    // here.
    //
    // It should be fine to create file manager here since the file system
    // options are stored in the compiler invocation and we can recreate the VFS
    // from the compiler invocation.
    if (!Clang->hasFileManager())
      Clang->createFileManager(createVFSFromCompilerInvocation(Clang->getInvocation(), Clang->getDiagnostics()));

    if (auto profilerOutput = Clang->createOutputFile(Clang->getFrontendOpts().TimeTracePath, /*Binary=*/false,
                                                      /*RemoveFileOnSignal=*/false,
                                                      /*useTemporary=*/false)) {
      llvm::timeTraceProfilerWrite(*profilerOutput);
      profilerOutput.reset();
      llvm::timeTraceProfilerCleanup();
      Clang->clearOutputFiles(false);
    }
  }

  // Our error handler depends on the Diagnostics object, which we're
  // potentially about to delete. Uninstall the handler now so that any
  // later errors use the default handling behavior instead.
  llvm::remove_fatal_error_handler();

  // When running with -disable-free, don't do any destruction or shutdown.
  if (Clang->getFrontendOpts().DisableFree) {
    llvm::BuryPointer(std::move(Clang));
  }
  return code;
}

bool polyregion::polystl::executeFrontendAction(clang::CompilerInstance *Clang, std::unique_ptr<clang::FrontendAction> Act) {
  // Honor -version.
  //
  // FIXME: Use a better -version message?
  if (Clang->getFrontendOpts().ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return true;
  }

  Clang->LoadRequestedPlugins();

  // Honor -mllvm.
  //
  // FIXME: Remove this, one day.
  // This should happen AFTER plugins have been loaded!
  if (!Clang->getFrontendOpts().LLVMArgs.empty()) {
    unsigned NumArgs = Clang->getFrontendOpts().LLVMArgs.size();
    auto Args = std::make_unique<const char *[]>(NumArgs + 2);
    Args[0] = "clang (LLVM option parsing)";
    for (unsigned i = 0; i != NumArgs; ++i)
      Args[i + 1] = Clang->getFrontendOpts().LLVMArgs[i].c_str();
    Args[NumArgs + 1] = nullptr;
    llvm::cl::ParseCommandLineOptions(NumArgs + 1, Args.get());
  }

#if CLANG_ENABLE_STATIC_ANALYZER
  // These should happen AFTER plugins have been loaded!

  AnalyzerOptions &AnOpts = *Clang->getAnalyzerOpts();

  // Honor -analyzer-checker-help and -analyzer-checker-help-hidden.
  if (AnOpts.ShowCheckerHelp || AnOpts.ShowCheckerHelpAlpha || AnOpts.ShowCheckerHelpDeveloper) {
    ento::printCheckerHelp(llvm::outs(), *Clang);
    return true;
  }

  // Honor -analyzer-checker-option-help.
  if (AnOpts.ShowCheckerOptionList || AnOpts.ShowCheckerOptionAlphaList || AnOpts.ShowCheckerOptionDeveloperList) {
    ento::printCheckerConfigList(llvm::outs(), *Clang);
    return true;
  }

  // Honor -analyzer-list-enabled-checkers.
  if (AnOpts.ShowEnabledCheckerList) {
    ento::printEnabledCheckerList(llvm::outs(), *Clang);
    return true;
  }

  // Honor -analyzer-config-help.
  if (AnOpts.ShowConfigOptionsList) {
    ento::printAnalyzerConfigList(llvm::outs());
    return true;
  }
#endif

  // If there were errors in processing arguments, don't do anything else.
  if (Clang->getDiagnostics().hasErrorOccurred()) return false;
  // Create and execute the frontend action.

  if (!Act) return false;
  bool Success = Clang->ExecuteAction(*Act);
  if (Clang->getFrontendOpts().DisableFree) llvm::BuryPointer(std::move(Act));
  return Success;
}

std::unique_ptr<clang::DiagnosticsEngine> polyregion::polystl::initialiseAndCreateDiag( //
    int argc, const char *argv[], const char *bugReportText) {
  llvm::InitLLVM init(argc, argv);
  llvm::setBugReportMsg(bugReportText);

  if (llvm::sys::Process::FixupStandardFileDescriptors()) return {};

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  llvm::SmallVector<const char *, 256> args(argv, argv + argc);

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts = clang::CreateAndPopulateDiagOpts(args);
  auto *diagClient = new clang::TextDiagnosticPrinter(llvm::errs(), &*diagOpts);
  diagClient->setPrefix(args[0]);
  auto diags = std::make_unique<clang::DiagnosticsEngine>(llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs()),
                                                          &*diagOpts, diagClient);
  if (!diagOpts->DiagnosticSerializationFile.empty()) {
    auto SerializedConsumer =
        clang::serialized_diags::create(diagOpts->DiagnosticSerializationFile, &*diagOpts, /*MergeChildRecords=*/true);
    diags->setClient(new clang::ChainedDiagnosticConsumer(diags->takeClient(), std::move(SerializedConsumer)));
  }
  clang::ProcessWarningOptions(*diags, *diagOpts, /*ReportDiags=*/false);
  return diags;
}
