#include "frontend.h"
#include <iostream>
#include <utility>

using namespace polyregion::polystl;

ModifyASTAndEmitObjAction::ModifyASTAndEmitObjAction( //
    decltype(mkConsumer) mkConsumer, decltype(endActionCb) endActionCb)
    : clang::EmitLLVMOnlyAction(), mkConsumer(std::move(mkConsumer)), endActionCb(std::move(endActionCb)) {}

std::unique_ptr<clang::ASTConsumer> ModifyASTAndEmitObjAction::CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef name) {
  std::vector<std::unique_ptr<clang::ASTConsumer>> xs;
  xs.emplace_back(mkConsumer());
  xs.emplace_back(clang::EmitLLVMOnlyAction::CreateASTConsumer(CI, name));
  return std::make_unique<clang::MultiplexConsumer>(std::move(xs));
}
void ModifyASTAndEmitObjAction::EndSourceFileAction() {
  std::cerr << "EndSourceFileAction pre" << std::endl;
  if (endActionCb) endActionCb();
  std::cerr << "EndSourceFileAction post" << std::endl;
  clang::EmitLLVMOnlyAction::EndSourceFileAction();
}
