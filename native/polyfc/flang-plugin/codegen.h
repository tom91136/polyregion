#pragma once

#include <string>
#include <vector>

#include "clang/AST/ASTContext.h"

#include "polyfront/options_backend.hpp"
#include "polyregion/types.h"

#include "remapper.h"

namespace polyregion::polyfc {

polyfront::KernelBundle compileRegion(clang::DiagnosticsEngine &diag, //
                                      const std::string &diagLoc,     //
                                      const polyfront::Options &opts, //
                                      runtime::PlatformKind kind,     //
                                      const std::string &moduleId,    //
                                      const Remapper::DoConcurrentRegion &region);
} // namespace polyregion::polyfc
