#pragma once

namespace polyregion::polystl {
int useCI(llvm::ArrayRef<const char *> Argv, const char *Argv0, void *MainAddr, const std::function<int(clang::CompilerInstance &)> &f);
bool executeFrontendAction(clang::CompilerInstance *Clang, std::unique_ptr<clang::FrontendAction> Act);
std::unique_ptr<clang::DiagnosticsEngine> initialiseAndCreateDiag(int argc, const char *argv[], const char *bugReportText);
} // namespace polyregion::polystl
