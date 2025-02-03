#pragma once

#include <string>
#include <vector>

#include "polyast.h"
#include "polyregion/types.h"

#include "polyfront/options_backend.hpp"

#include "clang/AST/ASTContext.h"

namespace polyregion::polystl {

polyfront::KernelBundle compileRegion(const polyfront::Options &ctx,
                            clang::ASTContext &C,                //
                            clang::DiagnosticsEngine &diag,      //
                            const std::string &moduleId,         //
                            const clang::CXXMethodDecl &functor, //
                            const clang::SourceLocation &loc,    //
                            runtime::PlatformKind kind);
} // namespace polyregion::polystl
