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

// the polyregion.core archive on the library path, if there is one
std::optional<polyfront::package::Archive> corePackage(clang::DiagnosticsEngine &diag, const polyfront::Options &opts);

// whether the archive has an implementation of `declaration` the capabilities allow
bool implements(const polyfront::package::Archive &archive, const std::string &declaration, const polyfront::Options &opts);

struct LinkedGroup {
  std::string suffix;
  polyast::CompileBundle compiled;
};

struct LinkedPackage {
  std::vector<LinkedGroup> groups;
  std::vector<std::string> implementations;
  std::vector<std::string> unfit;
};

struct PackageEntry {
  std::string entryName;
  const polyast::CompileBundle *compiled;
};

std::vector<PackageEntry> entriesFor(const LinkedPackage &linked, const std::string &entryName);

// links `root` against the archive's implementations of its declaration; unfit when the archive has none the capabilities allow
polyfront::package::Checked<LinkedPackage>
linkPackageProgram(const polyfront::Options &opts, const polyfront::package::Archive &archive, const polyast::Function &root,
                   std::vector<polyast::Function> consumerFunctions, std::vector<polyast::StructDef> consumerDefs,
                   std::vector<polyast::ProgramTypeSize> typeSizes, std::vector<std::string> &bitcodeFiles);

// calls the linked package entries at the builder's insertion point inside an acquired runtime context, loading their remote modules
// first and running the entry whose modules the device accepts; with a fallback, a missing context or module skips the call instead of
// exiting, and the returned i1 says whether it ran
mlir::Value emitPackageCall(mlir::OpBuilder &builder, mlir::ModuleOp &module, mlir::Location loc, const std::vector<mlir::Value> &args,
                            const std::vector<PackageEntry> &entries, bool fallback);

} // namespace polyregion::polyfc::interface_resolution
