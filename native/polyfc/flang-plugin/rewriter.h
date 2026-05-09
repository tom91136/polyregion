#pragma once

#include "clang/Basic/Diagnostic.h"
#include "mlir/IR/BuiltinOps.h"

namespace polyregion::polyfc {

void rewriteFIR(clang::DiagnosticsEngine &diag, mlir::ModuleOp &m);
void rewriteHLFIR(clang::DiagnosticsEngine &diag, mlir::ModuleOp &m);

} // namespace polyregion::polyfc