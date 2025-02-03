#include <iostream>

#include "aspartame/all.hpp"
#include "codegen.h"
#include "utils.h"

using namespace polyregion;
using namespace aspartame;


polyfront::KernelBundle polyfc::compileRegion( //
    clang::DiagnosticsEngine &diag, const std::string &diagLoc, const polyfront::Options &opts, runtime::PlatformKind kind,
    const std::string &moduleId, const Remapper::DoConcurrentRegion &region) {
  using Level = clang::DiagnosticsEngine::Level;
  const auto objects =
      opts.targets                                                                             //
      | filter([&](auto target, auto) { return runtime::targetPlatformKind(target) == kind; }) //
      | collect([&](auto &target, auto &features) {                                            //
          return polyfront::compileProgram(opts, region.program, target, features)             //
                 ^ fold_total([&](const polyast::CompileResult &r) -> std::optional<polyast::CompileResult> { return r; },
                              [&](const std::string &err) -> std::optional<polyast::CompileResult> {
                                emit(diag, Level::Warning, //
                                     "%0 [PolyDCO] Frontend failed to compile program [%1, target=%2, features=%3]\n%4", diagLoc, moduleId,
                                     std::string(to_string(target)), features, err);
                                return std::nullopt;
                              }) //
                 ^ map([&](auto &x) { return std::tuple{target, features, x}; });
        }) //
      | collect([&](auto &target, auto &features, auto &result) -> std::optional<polyfront::KernelObject> {
          auto targetName = std::string(to_string(target));
          emit(diag, Level::Remark, "%0 [PolyDCO] Compilation events for [%1, target=%2, features=%3]\n%4", //
               diagLoc, moduleId, targetName, features, repr(result));
          if (auto bin = result.binary) {
            auto size = std::to_string(static_cast<float>(bin->size()) / 1000.f);
            if (!result.messages.empty())
              emit(diag, Level::Warning, "%0 [PolyDCO] Backend emitted binary (%1KB) with warnings [%2, target=%3, features=%4]\n%5", //
                   diagLoc, size, moduleId, targetName, features, result.messages);
            else
              emit(diag, Level::Remark, "%0 [PolyDCO] Backend emitted binary (%1KB) [%2, target=%3, features=%4]", //
                   diagLoc, size, moduleId, targetName, features);
            if (auto format = runtime::moduleFormatOf(target)) {
              return polyfront::KernelObject{
                  *format,                                                                                                         //
                  *format == runtime::ModuleFormat::Object ? runtime::PlatformKind::HostThreaded : runtime::PlatformKind::Managed, //
                  result.features,                                                                                                 //
                  std::string(bin->begin(), bin->end())                                                                            //
              };
            } else
              emit(diag, Level::Remark, "%0 [PolyDCO] Backend emitted binary for unknown target [%1, target=%2,features=%3]", //
                   diagLoc, moduleId, targetName, features, result.messages);
          } else
            emit(diag, Level::Warning, "%0 [PolyDCO] Backend failed to compile program [%1, target=%2, features=%3]\nReason: %4", //
                 diagLoc, moduleId, targetName, features, result.messages);

          return std::nullopt;
        }) //
      | to_vector();
  return polyfront::KernelBundle{moduleId, objects, region.layouts, polyast::program_to_json(region.program).dump()};
}