#include "frontend.h"
#include <iostream>
#include <utility>

using namespace polyregion::polystl;

ModifyASTAndEmitObjAction::ModifyASTAndEmitObjAction(std::unique_ptr<FrontendAction> wrapped, //
                                                     decltype(mkConsumer) mkConsumer, decltype(endActionCb) endActionCb)
    : clang::WrapperFrontendAction(std::move(wrapped)), mkConsumer(std::move(mkConsumer)), endActionCb(std::move(endActionCb)) {}

std::unique_ptr<clang::ASTConsumer> ModifyASTAndEmitObjAction::CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef name) {
  std::vector<std::unique_ptr<clang::ASTConsumer>> xs;
  xs.emplace_back(mkConsumer());
  xs.emplace_back(clang::WrapperFrontendAction::CreateASTConsumer(CI, name));
  return std::make_unique<clang::MultiplexConsumer>(std::move(xs));
}
