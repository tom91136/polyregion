#pragma once

#include <string>
#include <vector>

#include "plugin.h"
#include "polyast.h"
#include "polyregion/types.h"

#include "clang/AST/ASTContext.h"

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
  std::vector<std::pair<bool, StructLayout>> layouts{};
  std::string metadata;
};

KernelBundle generateBundle(const Options &ctx,
                            clang::ASTContext &C,                //
                            clang::DiagnosticsEngine &diag,      //
                            const std::string &moduleId,         //
                            const clang::CXXMethodDecl &functor, //
                            const clang::SourceLocation &loc,    //
                            runtime::PlatformKind kind);
} // namespace polyregion::polystl
