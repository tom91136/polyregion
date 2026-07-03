// AUTO-GENERATED from PolyAST.Conventions via polyregion.ast.CodeGen. DO NOT EDIT.
#pragma once

#define POLYREFLECT_TRACK_ANNOTATION "polyreflect-track"
#define POLYREFLECT_RT_PROTECT_ANNOTATION "polyreflect-rt-protect"
#define POLYREFLECT_RT_ODR_ANNOTATION "polyreflect-rt-odr"
#define POLYREGION_LOCAL_ANNOTATION "__polyregion_local"

#define POLYREGION_RUNTIME_ABI(X)                                                                                                          \
  X(SmaAlloc, polyrt_sma_alloc)                                                                                                            \
  X(SmaEnsure, polyrt_sma_ensure)                                                                                                          \
  X(SmaEnsureMin, polyrt_sma_ensure_min)                                                                                                   \
  X(SmaEnsureDeep, polyrt_sma_ensure_deep)                                                                                                 \
  X(SmaPointeeSize, polyrt_sma_pointee_size)                                                                                               \
  X(SmaPatch, polyrt_sma_patch)                                                                                                            \
  X(SmaReadAlloc, polyrt_sma_read_alloc)                                                                                                   \
  X(SmaReadDeep, polyrt_sma_read_deep)                                                                                                     \
  X(SmaVisitClear, polyrt_sma_visit_clear)                                                                                                 \
  X(SmaMirrorGraph, polyrt_sma_mirror_graph)                                                                                               \
  X(SmaReadGraph, polyrt_sma_read_graph)                                                                                                   \
  X(SmaPoolReset, polyrt_sma_pool_reset)                                                                                                   \
  X(SmaPoolPtr, polyrt_sma_pool_ptr)

namespace polyregion::conventions {

inline constexpr auto EntryName = "_main";
inline constexpr auto ThisReceiver = "#this";
inline constexpr auto CaptureArg = "#capture";
inline constexpr auto BaseFieldPrefix = "#base";
inline constexpr auto EmptyStructStorageField = "#empty_struct_storage";
inline constexpr auto KernelBundleType = "KernelBundle";
inline constexpr auto AssertMessageLimit = 1024;

namespace reflect {
inline constexpr auto MirrorBitcodeGlobal = "polyregion_mirror_bc";
inline constexpr auto MirrorPrelude = "__polyregion_mirror_prelude";
inline constexpr auto MirrorPostlude = "__polyregion_mirror_postlude";
inline constexpr auto FlagVerbose = "polyreflect-verbose";
inline constexpr auto FlagEarly = "polyreflect-early";
inline constexpr auto FlagLate = "polyreflect-late";
inline constexpr auto PassRecordAlloc = "polyreflect-record-alloc";
inline constexpr auto PassStack = "polyreflect-stack";
inline constexpr auto PassMem = "polyreflect-mem";
inline constexpr auto PassLinkMirror = "polyreflect-link-mirror";
inline constexpr auto PassProtectRt = "polyreflect-protect-rt";
inline constexpr auto PassInterpose = "polyreflect-interpose";
} // namespace reflect

} // namespace polyregion::conventions
