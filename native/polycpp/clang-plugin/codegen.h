#pragma once

#include <string>
#include <vector>

#include "clang/AST/ASTContext.h"

#include "polyfront/options_backend.hpp"
#include "polyregion/types.h"

#include "polyast.h"

namespace polyregion::polystl {

struct PackageExport {
  const clang::FunctionDecl *decl;
  polyregion::polyast::Sym name;
};

polyfront::KernelBundle compileRegion(const polyfront::Options &ctx,
                                      clang::ASTContext &C,                //
                                      clang::DiagnosticsEngine &diag,      //
                                      const std::string &moduleId,         //
                                      const clang::CXXMethodDecl &functor, //
                                      const clang::SourceLocation &loc,    //
                                      runtime::PlatformKind kind);

void compilePackageProgram(const polyfront::Options &opts,            //
                           clang::ASTContext &C,                      //
                           clang::DiagnosticsEngine &diag,            //
                           const std::vector<PackageExport> &exports, //
                           const std::string &outPath);
} // namespace polyregion::polystl
