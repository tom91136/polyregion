#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "polyregion/export.h"
#include "polyregion/types.h"

#include "ast.h"

namespace polyregion::compiler {

using MonoClock = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;

[[nodiscard]] TimePoint nowMono();
[[nodiscard]] int64_t elapsedNs(const TimePoint &a, const TimePoint &b = nowMono());
[[nodiscard]] int64_t nowMs();

POLYREGION_EXPORT void initialise();

inline constexpr auto DefaultPipelineSpec = "FullOpt;StructuredExit";
inline constexpr uint32_t DefaultWorkgroupMemoryBytes = 32768;

[[nodiscard]] constexpr uint32_t defaultWorkgroupMemoryBytes(const compiletime::Target target) {
  switch (target) {
    case compiletime::Target::Object_LLVM_AMDGCN: return 65536;
    case compiletime::Target::Object_LLVM_NVPTX64: return 49152;
    default: return DefaultWorkgroupMemoryBytes;
  }
}

struct POLYREGION_EXPORT Options {
  POLYREGION_EXPORT compiletime::Target target = compiletime::Target::Object_LLVM_HOST;
  POLYREGION_EXPORT std::string arch;
  POLYREGION_EXPORT std::string pipelineSpec = {}; // pipeline spec ("Name(k=v,k=v); Name; ...")
  POLYREGION_EXPORT bool hostMirroring = false;    // compile only Host-affinity functions, emit bitcode
  POLYREGION_EXPORT uint32_t workgroupMemoryBytes = defaultWorkgroupMemoryBytes(target);
};

struct POLYREGION_EXPORT Staged {
  POLYREGION_EXPORT std::optional<polyast::Program> program;
  POLYREGION_EXPORT Options options;
  POLYREGION_EXPORT bool hadEntry = false;
  POLYREGION_EXPORT std::vector<polyast::CompileEvent> events;
  POLYREGION_EXPORT std::string error;
};

POLYREGION_EXPORT std::vector<polyast::StructLayout> layoutOf(const std::vector<polyast::StructDef> &sdefs, const Options &options);
POLYREGION_EXPORT std::vector<polyast::StructLayout> layoutOf(const polyast::Bytes &bytes, const Options &options);

POLYREGION_EXPORT polyast::Program runPipeline(const polyast::Program &program, const std::string &spec = {});
// stage runs the PolyPass pipeline and must stay on one thread; emit is the code generator and may run concurrently
POLYREGION_EXPORT Staged stage(const polyast::Program &program, const Options &options);
POLYREGION_EXPORT Staged stage(polyast::Program &&program, const Options &options);
POLYREGION_EXPORT polyast::CompileResult emit(Staged staged, const compiletime::OptLevel &opt);
POLYREGION_EXPORT polyast::CompileResult compile(const polyast::Program &program, const Options &options, const compiletime::OptLevel &opt);
POLYREGION_EXPORT polyast::CompileResult compile(const polyast::Bytes &astBytes, const Options &options, const compiletime::OptLevel &opt);

} // namespace polyregion::compiler
