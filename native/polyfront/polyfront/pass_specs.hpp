#pragma once

#include <string>
#include <vector>

#include "fmt/format.h"

#include "polyregion/types.h"

namespace polyregion::polyfront::passes {

// single-arena lowering: the dispatch marshals the capture graph into one device arena (pointers -> i64
// offsets) and the pass rewrites every opaque-pointer deref. flat backends (c_source/OpenCL 1.2) use byte
// addressing `(T*)&arena[off]`; SPIR-V cannot cast int<->ptr, so it reads through typed scalar views
inline std::string deviceArena() { return "FullOpt;ArenaLower;VerifyAnchors(strict=true)"; }
// VerifyAnchors(strict) after ArenaView asserts every opaque-origin access resolved to an arena view
// (logical SPIR-V cannot deref a raw pointer) - a missed deref becomes a compile error, not a device fault
inline std::string deviceArenaLogical() { return "FullOpt;Anchor;ArenaView;RegionRespace;VerifyAnchors(strict=true)"; }

inline std::string hostMirror(const std::string &mirrorId) { return fmt::format("Mirror(id={})", mirrorId); }

// binding-slot targets use the arena (byte addressing on flat c_source, typed views on SPIR-V);
// physical backends (PTX/HSACO) and host get no arena pass and marshal via the compile-time mirror
inline std::vector<std::string> arenaPassesFor(const compiletime::Target &target) {
  switch (target) {
    case compiletime::Target::Object_LLVM_SPIRV_GLCompute:
    case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
    case compiletime::Target::Object_LLVM_SPIRV64_Kernel: return {"--passes", deviceArenaLogical()};
    case compiletime::Target::Source_C_OpenCL1_1:
    case compiletime::Target::Source_C_Metal1_0: return {"--passes", deviceArena()};
    default: return {};
  }
}

} // namespace polyregion::polyfront::passes
