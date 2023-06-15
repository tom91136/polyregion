#include "frontend.h"
#include <iostream>

using namespace polyregion::polystl;

RewriteAndCompileAction::RewriteAndCompileAction(decltype(downstreamAction) downstreamAction,
                                                 const decltype(createConsumer) &createConsumer)
    : downstreamAction(std::move(downstreamAction)), createConsumer(createConsumer) {}

std::unique_ptr<clang::ASTConsumer> RewriteAndCompileAction::CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef name) {
  compilerInstance = &CI;
  fileName = name.str();
  rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
  return createConsumer(rewriter, skipEndAction);
}

clang::PluginASTAction::ActionType RewriteAndCompileAction::getActionType() { return clang::PluginASTAction::ReplaceAction; }
bool RewriteAndCompileAction::ParseArgs(const clang::CompilerInstance &, const std::vector<std::string> &) { return true; }
void RewriteAndCompileAction::EndSourceFileAction() {
  std::cerr << "End Action" << std::endl;
  if (skipEndAction) return;

  auto &cliArgs = compilerInstance->getCodeGenOpts().CommandLineArgs;
  std::vector<const char *> constCliArgs(cliArgs.size());
  std::transform(cliArgs.begin(), cliArgs.end(), constCliArgs.begin(), [](auto &s) { return s.c_str(); });
  auto CInvNew = std::make_shared<clang::CompilerInvocation>();
  bool CInvNewCreated = clang::CompilerInvocation::CreateFromArgs(*CInvNew, constCliArgs, compilerInstance->getDiagnostics());
  assert(CInvNewCreated);
  auto &Target = compilerInstance->getTarget();

  clang::CompilerInstance CINew;
  CINew.setInvocation(CInvNew);
  CINew.setTarget(&Target);
  CINew.createDiagnostics();

  if (auto buffer = rewriter.getRewriteBufferFor(compilerInstance->getSourceManager().getMainFileID()); buffer) {
    std::string content = {buffer->begin(), buffer->end()};
    std::cerr << "====" << fileName << "====" << std::endl;
    std::cerr << content << std::endl;
    std::cerr << "=========" << std::endl;
    CINew.getPreprocessorOpts().addRemappedFile(fileName, llvm::MemoryBuffer::getMemBufferCopy(content).release());
  } else {
    std::cerr << "Buffer unchanged" << std::endl;
  }

  if (downstreamAction) CINew.ExecuteAction(*downstreamAction);
}
