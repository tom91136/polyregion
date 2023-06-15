#include <cstdlib>
#include <iostream>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
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
#include "outliner.h"

int main(int argc, const char *argv[]) {

  std::vector<std::string> actual;
  {
    std::string triple = llvm::sys::getDefaultTargetTriple();
    auto diagOptions = new clang::DiagnosticOptions();
    auto diagClient = new clang::TextDiagnosticPrinter(llvm::errs(), &*diagOptions);
    auto diagID = new clang::DiagnosticIDs();
    auto diags = clang::DiagnosticsEngine(diagID, diagOptions, diagClient);

    auto execPath = llvm::sys::fs::getMainExecutable(argv[0], nullptr);
    auto execParentDir = llvm::sys::path::parent_path(execPath).str();

    clang::driver::Driver D(execPath, triple, diags);
    D.ResourceDir = execParentDir + "/lib/clang/" + std::to_string(CLANG_VERSION_MAJOR);
    std::unique_ptr<clang::driver::Compilation> C(D.BuildCompilation(llvm::ArrayRef(argv, argc)));

    auto job = *C->getJobs().begin();
    for (auto arg : job.getArguments())
      actual.push_back(arg);
  }
  actual.erase(actual.begin()); // drop the initial -cc1

  std::vector<const char *> cc1argsCstrs;
  for (const auto &arg : actual) {
    cc1argsCstrs.push_back(arg.c_str());
    std::cout << arg << " ";
  }
  std::cout << std::endl;

  auto diagOptions = clang::CreateAndPopulateDiagOpts(cc1argsCstrs);
  auto diagClient = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), &*diagOptions);
  auto diag = clang::CompilerInstance::createDiagnostics(diagOptions.release(), diagClient.release(), true);

  //  auto CInvNew = std::make_shared<clang::CompilerInvocation>();

  clang::CompilerInstance CINew;
  CINew.setDiagnostics(diag.get());
  bool CInvNewCreated = clang::CompilerInvocation::CreateFromArgs(CINew.getInvocation(), cc1argsCstrs, *diag);
  assert(CInvNewCreated);
  CINew.setTarget(clang::TargetInfo::CreateTargetInfo(CINew.getDiagnostics(), CINew.getInvocation().TargetOpts));
  CINew.createSourceManager(*CINew.createFileManager());
  CINew.createPreprocessor(clang::TU_Complete);
  CINew.createASTContext();

  //  auto opt = CINew.getInvocation().getFrontendOpts();
  //  auto that = CINew.getFileManager().getFile(opt.Inputs[0].getFile());
  //
  //  clang::FileID mainFileID = CINew.getSourceManager().createFileID(that.get(), clang::SourceLocation(), clang::SrcMgr::C_User);
  //  CINew.getSourceManager().setMainFileID(mainFileID);
  //
  //  CINew.getDiagnosticClient().BeginSourceFile(CINew.getLangOpts());
  //

  using namespace polyregion::polystl;

  struct ToolRewriteAndCompileAction : public RewriteAndCompileAction {
    ToolRewriteAndCompileAction()
        : RewriteAndCompileAction({},
                                  [](clang::Rewriter &r, std::atomic_bool &error) { return std::make_unique<OutlineConsumer>(r, error); }) {
    }
  };

  //  clang::ParseAST(CINew.getPreprocessor(), cc, CINew.getASTContext());
  //  std::cout << "|||"
  //            << "\n";
  //
  //  delete cc;
  //  std::cout << "---"
  //            << "\n";
  ToolRewriteAndCompileAction act;
  CINew.ExecuteAction(act);

  //  CINew.createASTContext();
  //  CINew.createASTReader();
  //  auto reader = CINew.getASTReader().get();

  //  CINew.getSourceManager().PrintStats();
  //  auto theFIle = CINew.getSourceManager().getBufferData(CINew.getSourceManager().getMainFileID());
  //  std::cout <<theFIle.str() << std::endl;

  //  auto r = reader->ReadAST("hello.cpp", clang::serialization::MK_MainFile,
  //                  clang::SourceLocation(),
  //                  clang::ASTReader::ARR_None);
  //  std::cout << r << std::endl;

  // create "virtual" input file
  //  auto &PreprocessorOpts = CINew.getPreprocessorOpts();
  //  CINew.setASTConsumer();

  std::cout << "O=" << CINew.getFileManager().getNumUniqueRealFiles() << " ok" << CInvNewCreated << std::endl;

  return EXIT_SUCCESS;
}