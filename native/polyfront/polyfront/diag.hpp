#pragma once

#include <cstddef>
#include <utility>

namespace polyregion::polyfront {

#define POLYREGION_DIAG_POLYSTL "[PolySTL] "
#define POLYREGION_DIAG_POLYCPP "[PolyCpp] "
#define POLYREGION_DIAG_POLYFC "[PolyFC] "
#define POLYREGION_DIAG_POLYDCO "[PolyDCO] "

template <typename Diag, std::size_t N, typename... Args>
void emit(Diag &diag, typename Diag::Level level, const char (&fmt)[N], Args &&...args) {
  auto b = diag.Report(diag.getCustomDiagID(level, fmt));
  (b << ... << std::forward<Args>(args));
}

template <typename Diag, typename Loc, std::size_t N, typename... Args>
void emit(Diag &diag, Loc loc, typename Diag::Level level, const char (&fmt)[N], Args &&...args) {
  auto b = diag.Report(loc, diag.getCustomDiagID(level, fmt));
  (b << ... << std::forward<Args>(args));
}

} // namespace polyregion::polyfront
