#pragma once

#include "options.h"
#include "types.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

namespace polyregion::polystl {

struct KernelObject {
  runtime::ModuleFormat format{};
  runtime::PlatformKind kind{};
  std::vector<std::string> features{};
  std::string moduleImage{};
};

struct KernelBundle {
  std::string moduleName{};
  std::vector<KernelObject> objects{};
  std::string metadata;
};

KernelBundle generate(clang::ASTContext &C,                //
                      clang::DiagnosticsEngine &diag,      //
                      const std::string &moduleId,         //
                      const clang::CXXMethodDecl &functor, //
                      const clang::SourceLocation &loc,
                      runtime::PlatformKind kind, //
                      const StdParOptions &opts);
}
