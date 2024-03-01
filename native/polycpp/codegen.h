#pragma once

#include "types.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

namespace polyregion::polystl {
polyregion::runtime::KernelBundle generate(clang::ASTContext &C,                //
                                           clang::DiagnosticsEngine &diag,      //
                                           const std::string &moduleId,         //
                                           const clang::CXXMethodDecl &functor, //
                                           const std::vector<std::pair<compiletime::Target, std::string>> &targets);
}
