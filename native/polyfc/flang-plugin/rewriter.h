#pragma once

#include <string>
#include <vector>

#include "clang/Basic/Diagnostic.h"
#include "mlir/IR/BuiltinOps.h"

namespace polyregion::polyfc {

void rewriteFIR(clang::DiagnosticsEngine &diag, mlir::ModuleOp &m, std::vector<std::string> &bitcodeFiles);
void rewriteHLFIR(clang::DiagnosticsEngine &diag, mlir::ModuleOp &m);

} // namespace polyregion::polyfc
