#include "codegen.h"

#include "flang/Optimizer/Dialect/FIROpsSupport.h"

#include "aspartame/all.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyfront/diag.hpp"
#include "polyfront/library_emit.hpp"
#include "polyfront/pass_specs.hpp"
#include "polyregion/env_keys.h"

#include "ast.h"
#include "utils.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace aspartame;

polyfront::KernelBundle polyfc::compileRegion( //
    clang::DiagnosticsEngine &diag, const std::string &diagLoc, const polyfront::Options &opts, runtime::PlatformKind kind,
    const std::string &moduleId, const Remapper::DoConcurrentRegion &region) {
  using Level = clang::DiagnosticsEngine::Level;
  if (opts.jit) {
    auto jitObjects =
        opts.targets                                                                                            //
        | filter([&](const auto &target, const auto &) { return runtime::targetPlatformKind(target) == kind; }) //
        | collect([&](const auto &target, const auto &arch) -> std::optional<polyfront::KernelObject> {
            auto format = runtime::moduleFormatOf(target);
            if (!format) return std::nullopt;
            const auto pp = polyfront::passes::arenaPassesFor(target, opts.stackDepth);
            polyfront::KernelObject ko;
            ko.format = *format;
            ko.kind = *format == runtime::ModuleFormat::Object ? runtime::PlatformKind::HostThreaded : runtime::PlatformKind::Managed;
            ko.features = polyfront::passes::jitFeaturesFor(target);
            ko.target = target;
            ko.arch = arch;
            ko.pipelineSpec = pp.size() >= 2 ? pp[1] : std::string{};
            return ko;
          }) //
        | to_vector();
    const auto packed = polyast::hashed_program_to_msgpack(region.program);
    const bool jitAsserts = !region.program.template collect_all<polyast::Spec::Assert>().empty();
    return polyfront::KernelBundle{moduleId,
                                   jitObjects,
                                   region.layouts,
                                   {},
                                   polyast::program_to_json(region.program).dump(),
                                   {},
                                   {},
                                   jitAsserts,
                                   std::string(packed.begin(), packed.end())};
  }
  const auto compiled =
      opts.targets                                                                                            //
      | filter([&](const auto &target, const auto &) { return runtime::targetPlatformKind(target) == kind; }) //
      | collect([&](const auto &target, const auto &features) {                                               //
          return polyfront::compileProgram(opts, region.program, target, features,
                                           polyfront::passes::arenaPassesFor(target, opts.stackDepth)) //
                 ^ fold_total([&](const polyast::CompileResult &r) -> std::optional<polyast::CompileResult> { return r; },
                              [&](const std::string &err) -> std::optional<polyast::CompileResult> {
                                emit(diag, Level::Warning, //
                                     "%0 " POLYREGION_DIAG_POLYDCO "Frontend failed to compile program [%1, target=%2, features=%3]\n%4",
                                     diagLoc, moduleId, std::string(magic_enum::enum_name(target)), features, err);
                                return std::nullopt;
                              }) //
                 ^ map([&](const auto &x) { return std::tuple{target, features, x}; });
        }) //
      | to_vector();

  const bool asserts =
      compiled ^ exists([](const auto &, const auto &, const auto &result) { return polyfront::entryNeedsErrorBuffer(result); });

  const auto objects =
      compiled | collect([&](const auto &target, const auto &features, const auto &result) -> std::optional<polyfront::KernelObject> {
        auto targetName = std::string(magic_enum::enum_name(target));
        emit(diag, Level::Remark, "%0 " POLYREGION_DIAG_POLYDCO "Compilation events for [%1, target=%2, features=%3]\n%4", //
             diagLoc, moduleId, targetName, features, repr(result));
        if (auto bin = result.binary) {
          auto size = std::to_string(static_cast<float>(bin->size()) / 1000.f);
          if (!result.messages.empty())
            emit(diag, Level::Warning,
                 "%0 " POLYREGION_DIAG_POLYDCO "Backend emitted binary (%1KB) with warnings [%2, target=%3, features=%4]\n%5", //
                 diagLoc, size, moduleId, targetName, features, result.messages);
          else
            emit(diag, Level::Remark, "%0 " POLYREGION_DIAG_POLYDCO "Backend emitted binary (%1KB) [%2, target=%3, features=%4]", //
                 diagLoc, size, moduleId, targetName, features);
          if (auto format = runtime::moduleFormatOf(target)) {
            return polyfront::KernelObject{
                *format,                                                                                                         //
                *format == runtime::ModuleFormat::Object ? runtime::PlatformKind::HostThreaded : runtime::PlatformKind::Managed, //
                result.features,                                                                                                 //
                std::string(bin->begin(), bin->end())                                                                            //
            };
          } else
            emit(diag, Level::Remark,
                 "%0 " POLYREGION_DIAG_POLYDCO "Backend emitted binary for unknown target [%1, target=%2,features=%3]", //
                 diagLoc, moduleId, targetName, features, result.messages);
        } else
          emit(diag, Level::Warning,
               "%0 " POLYREGION_DIAG_POLYDCO "Backend failed to compile program [%1, target=%2, features=%3]\nReason: %4", //
               diagLoc, moduleId, targetName, features, result.messages);

        return std::nullopt;
      }) //
      | to_vector();
  // If targets were requested for this kind but every one failed, escalate to a hard error so
  // the user sees the failure at compile time instead of an opaque "no compatible image" abort
  // from the runtime later on.
  const auto requestedForKind =
      opts.targets ^ count([&](const auto &target, const auto &) { return runtime::targetPlatformKind(target) == kind; });
  if (requestedForKind > 0 && objects.empty()) {
    emit(diag, Level::Error,
         "%0 " POLYREGION_DIAG_POLYDCO
         "No kernels compiled successfully for [%1, kind=%2] (requested %3 target(s)); see prior diagnostics for the "
         "per-target "
         "failure",
         diagLoc, moduleId, std::string(magic_enum::enum_name(kind)), std::to_string(static_cast<int>(requestedForKind)));
  }
  auto mir = polyfront::compileManagedHostMirror(opts, region.program, kind, moduleId);
  if (mir.error)
    emit(diag, Level::Warning, "%0 " POLYREGION_DIAG_POLYDCO "Host mirroring compile failed [%1]: %2", diagLoc, moduleId, *mir.error);
  return polyfront::KernelBundle{
      moduleId,    objects,      region.layouts, /*readOnlyMembers*/ {}, polyast::program_to_json(region.program).dump(),
      mir.bitcode, mir.mirrorId, asserts};
}

void polyfc::compileLibrary(clang::DiagnosticsEngine &diag, const polyfront::Options &opts, mlir::ModuleOp &m, mlir::DataLayout &L,
                            const std::string &outPath) {
  using Level = clang::DiagnosticsEngine::Level;

  std::vector<mlir::func::FuncOp> exports;
  m.walk([&](mlir::func::FuncOp f) {
    if (!f.isExternal() && fir::hasBindcAttr(f.getOperation())) exports.push_back(f);
  });
  if (exports.empty()) emit(diag, Level::Warning, POLYREGION_DIAG_POLYDCO "-fstdpar-emit-library set but no bind(c) procedures found");

  Remapper r(m, L, m.getOperation(), {});
  size_t exported = 0;
  for (auto f : exports) {
    r.handleFunc(f);
    const auto name = f.getSymName().str();
    if (const auto it = r.userFuncs.find(name); it != r.userFuncs.end()) {
      it->second.visibility = FunctionVisibility::Exported();
      exported++;
      if (opts.verbose) emit(diag, Level::Remark, POLYREGION_DIAG_POLYDCO "Exporting library symbol: %0", name);
    }
  }

  const auto program = polyfront::libraryProgram(r.functions | concat(r.userFuncs | values()) | to_vector(),
                                                 r.defs | values() | concat(r.syntheticDefs) | to_vector());

  polyfront::writeProgramMsgpack(program, outPath) //
      ^ foreach_total(
          [&](const std::error_code &ec) {
            emit(diag, Level::Error, POLYREGION_DIAG_POLYDCO "Cannot open library output %0: %1", outPath, ec.message());
          },
          [&](const size_t bytes) {
            emit(diag, Level::Remark, POLYREGION_DIAG_POLYDCO "Wrote polyAST library %0 (%1 symbols, %2 functions, %3 bytes)", outPath,
                 std::to_string(exported), std::to_string(program.functions.size()), std::to_string(bytes));
          });
}
