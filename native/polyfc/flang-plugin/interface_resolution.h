#pragma once

#include <optional>
#include <string>
#include <vector>

#include "clang/Basic/Diagnostic.h"
#include "mlir/IR/BuiltinOps.h"

#include "polyfront/options_backend.hpp"
#include "polyfront/package.hpp"
#include "polyfront/polyc_client.hpp"

namespace polyregion::polyfc::interface_resolution {

void resolveInterfaces(clang::DiagnosticsEngine &diag, mlir::ModuleOp &module, const polyfront::Options &opts,
                       std::vector<std::string> &bitcodeFiles);

// links `root` against the archive's implementations of its declaration and keeps the host bitcode; an empty value means no
// implementation the capabilities allow, so the caller keeps its own lowering
polyfront::package::Checked<std::optional<polyast::CompileBundle>>
linkPackageProgram(const polyfront::Options &opts, const polyfront::package::Archive &archive, const polyast::Function &root,
                   std::vector<polyast::Function> consumerFunctions, std::vector<polyast::StructDef> consumerDefs,
                   std::vector<polyast::ProgramTypeSize> typeSizes, std::vector<std::string> &bitcodeFiles);

// calls a linked package entry at the builder's insertion point inside an acquired runtime context, loading its remote modules first
void emitPackageCall(mlir::OpBuilder &builder, mlir::ModuleOp &module, mlir::Location loc, const std::string &entryName,
                     const std::vector<mlir::Value> &args, const polyast::CompileBundle &compiled);

} // namespace polyregion::polyfc::interface_resolution
