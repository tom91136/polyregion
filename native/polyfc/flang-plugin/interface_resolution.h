#pragma once

#include <string>
#include <vector>

#include "clang/Basic/Diagnostic.h"
#include "mlir/IR/BuiltinOps.h"

#include "polyfront/options_backend.hpp"

namespace polyregion::polyfc::interface_resolution {

void resolveInterfaces(clang::DiagnosticsEngine &diag, mlir::ModuleOp &module, const polyfront::Options &opts,
                       std::vector<std::string> &bitcodeFiles);

}
