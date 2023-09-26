#pragma once

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

namespace polyregion::polyast {

std::string generate(clang::ASTContext &C, const clang::CXXRecordDecl *parent, clang::QualType returnTpe, const clang::Stmt *body);

}
