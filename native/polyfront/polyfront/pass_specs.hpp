#pragma once

#include <optional>
#include <string>
#include <vector>

#include "fmt/format.h"

#include "polyregion/types.h"

namespace polyregion::polyfront::passes {

// an unset depth leaves FullOpt argument-free, so the frame-stack default lives solely in FullOpt's Scala side
inline std::string fullOpt(std::optional<int> stackDepth) {
  return stackDepth ? fmt::format("FullOpt(stackDepth={})", *stackDepth) : std::string("FullOpt");
}

// single-arena lowering: the dispatch marshals the capture graph into one device arena (pointers -> i64
// offsets) and the pass rewrites every opaque-pointer deref. flat backends (c_source/OpenCL 1.2) use byte
// addressing `(T*)&arena[off]`; SPIR-V cannot cast int<->ptr, so it reads through typed scalar views.
// StructuredExit runs before ArenaLower so a runtime assert message's copy loop (an opaque-pointer deref) is
// arena-lowered too; it stays after FullOpt so the optimiser cannot DCE its error-buffer writes
inline std::string deviceArena(std::optional<int> stackDepth = {}) {
  return fullOpt(stackDepth) + ";StructuredExit;ArenaLower;VerifyAnchors(strict=true)";
}
// VerifyAnchors(strict) after ArenaView asserts every opaque-origin access resolved to an arena view
// (logical SPIR-V cannot deref a raw pointer) - a missed deref becomes a compile error, not a device fault.
// PartialEval(canonicaliseAddresses=true) is the address-canonicalisation-only mode (no fold/DCE) that
// root-anchors derived-pointer temps; it runs after StructuredExit so the temps that lowering injects are
// canonicalised too, without disturbing the assert/#error side-channel writes
inline std::string deviceArenaLogical(std::optional<int> stackDepth = {}) {
  return fullOpt(stackDepth) + ";StructuredExit;PartialEval(canonicaliseAddresses=true);ArenaView;RegionRespace;VerifyAnchors(strict=true)";
}

inline std::string hostMirror(const std::string &mirrorId) { return fmt::format("Mirror(id={})", mirrorId); }

// binding-slot targets use the arena (byte addressing on flat c_source, typed views on SPIR-V);
// physical backends (PTX/HSACO) and host get no arena pass and marshal via the compile-time mirror
inline std::vector<std::string> arenaPassesFor(const compiletime::Target &target, std::optional<int> stackDepth = {}) {
  switch (target) {
    case compiletime::Target::Object_LLVM_SPIRV_GLCompute:
    case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
    case compiletime::Target::Object_LLVM_SPIRV64_Kernel: return {"--passes", deviceArenaLogical(stackDepth)};
    case compiletime::Target::Source_C_OpenCL1_1:
    case compiletime::Target::Source_C_Metal1_0: return {"--passes", deviceArena(stackDepth)};
    default: return stackDepth ? std::vector<std::string>{"--passes", fullOpt(stackDepth) + ";StructuredExit"} : std::vector<std::string>{};
  }
}

} // namespace polyregion::polyfront::passes
