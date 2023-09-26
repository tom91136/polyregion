#include <cstdlib>
#include <iostream>
#include <llvm/Support/TargetSelect.h>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/Host.h"

#include "frontend.h"
#include "rewriter.h"

void executeCC1(const std::vector<const char *> &cc1argsCstrs) {
  auto diagOptions = clang::CreateAndPopulateDiagOpts(cc1argsCstrs);
  auto diagClient = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), &*diagOptions);
  auto diag = clang::CompilerInstance::createDiagnostics(diagOptions.release(), diagClient.release(), true);

  //  auto CInvNew = std::make_shared<clang::CompilerInvocation>();

  clang::CompilerInstance CINew;
  CINew.setDiagnostics(diag.get());
  bool CInvNewCreated = clang::CompilerInvocation::CreateFromArgs(CINew.getInvocation(), cc1argsCstrs, *diag);
  assert(CInvNewCreated);
  CINew.setTarget(clang::TargetInfo::CreateTargetInfo(CINew.getDiagnostics(), CINew.getInvocation().TargetOpts));
  //  CINew.createSourceManager(*CINew.createFileManager());
  //  CINew.createPreprocessor(clang::TU_Complete);
  //  CINew.createASTContext();
  //  CINew.createASTReader();

  //  auto opt = CINew.getInvocation().getFrontendOpts();
  //  auto that = CINew.getFileManager().getFile(opt.Inputs[0].getFile());
  //
  //  clang::FileID mainFileID = CINew.getSourceManager().createFileID(that.get(), clang::SourceLocation(), clang::SrcMgr::C_User);
  //  CINew.getSourceManager().setMainFileID(mainFileID);
  //
  //  CINew.getDiagnosticClient().BeginSourceFile(CINew.getLangOpts());
  //

  using namespace polyregion::polystl;
  ModifyASTAndEmitObjAction action([]() { return std::make_unique<OffloadRewriteConsumer>(); });
  auto ok = CINew.ExecuteAction(action);
  assert(ok);

  //  CINew.getASTContext().getTranslationUnitDecl()->getFirstDecl()->dumpColor();

  //  CINew.createASTContext();
  //  CINew.createASTReader();
  //  auto reader = CINew.getASTReader().get();

  CINew.getSourceManager().PrintStats();
  //  auto theFIle = CINew.getSourceManager().getBufferData(CINew.getSourceManager().getMainFileID());
  //  std::cout <<theFIle.str() << std::endl;

  //  auto r = reader->ReadAST("hello.cpp", clang::serialization::MK_MainFile,
  //                  clang::SourceLocation(),
  //                  clang::ASTReader::ARR_None);
  //  std::cout << r << std::endl;

  // create "virtual" input file
  auto &PreprocessorOpts = CINew.getPreprocessorOpts();
  //  CINew.setASTConsumer();

  std::cout << "O=" << CINew.getFileManager().getNumUniqueRealFiles() << " ok" << CInvNewCreated << std::endl;
}

int main(int argc, const char *argv[]) {

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

  clang::driver::Driver D(execPath, triple, diags);
  D.ResourceDir = execParentDir + "/lib/clang/" + std::to_string(CLANG_VERSION_MAJOR);

  std::vector<const char *> modifiedArgs(argv, argv + argc);
  // since the executable name won't be clang++ anymore, we manually set the mode to C++ by inserting the override after the executable name
  modifiedArgs.insert(std::next(modifiedArgs.begin()), "--driver-mode=g++");

  std::unique_ptr<clang::driver::Compilation> C(D.BuildCompilation(llvm::ArrayRef(modifiedArgs)));
  //  D.BuildJobs(*C);

  std::cout << "Jobs = " << C->getJobs().size() << "\n";
  for (const auto &command : C->getJobs()) {
    const auto &args = command.getArguments();

    std::cout << command.getExecutable() << " ";
    for (auto arg : args)
      std::cout << arg << " ";
    std::cout << std::endl;

    if (command.getExecutable() == execPath &&                    // make sure the driver is actually calling us
        command.getCreator().getName() == std::string("clang") && // and that clang is the compiler
        args[0] == std::string("-cc1")                            // and we're invoking the cc1 frontend
    ) {
      std::cerr << "Replacing cc1 invocation for " << command.getSource().getClassName() << "\n";
      std::cout << command.getExecutable() << " ";
      for (auto arg : args)
        std::cout << arg << " ";
      std::cout << std::endl;
      std::vector<const char *> actual;

      actual.insert(actual.begin(), std::next(args.begin()), args.end()); //  skip the first -cc1
      executeCC1(actual);
    } else {
      const clang::driver::Command *failed{};
      if (auto code = C->ExecuteCommand(command, failed); code != 0) {
        std::cerr << "Command exited with code " << code << ":\n";
        std::cout << command.getExecutable() << " ";
        for (auto arg : args)
          std::cout << arg << " ";
        std::cout << std::endl;
        std::exit(code);
      }
    }
  }
  return EXIT_SUCCESS;
}