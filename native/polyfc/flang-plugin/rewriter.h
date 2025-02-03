#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "clang/Basic/Diagnostic.h"

namespace polyregion::polyfc {

void rewriteFIR(clang::DiagnosticsEngine &diag, mlir::ModuleOp &m);
void rewriteHLFIR(clang::DiagnosticsEngine &diag, mlir::ModuleOp &m);

} // namespace polyregion::polyfc