#pragma once

#include <string>
#include <vector>

#include "clang/AST/ASTContext.h"

#include "polyfront/options_backend.hpp"
#include "polyregion/types.h"

#include "polyast.h"

namespace polyregion::polystl {

polyfront::KernelBundle compileRegion(const polyfront::Options &ctx,
                                      clang::ASTContext &C,                //
                                      clang::DiagnosticsEngine &diag,      //
                                      const std::string &moduleId,         //
                                      const clang::CXXMethodDecl &functor, //
                                      const clang::SourceLocation &loc,    //
                                      runtime::PlatformKind kind);

void compileLibrary(const polyfront::Options &opts,                          //
                    clang::ASTContext &C,                                    //
                    clang::DiagnosticsEngine &diag,                          //
                    const std::vector<const clang::FunctionDecl *> &exports, //
                    const std::string &outPath);
} // namespace polyregion::polystl
