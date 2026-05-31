#pragma once

namespace polyregion::msvc_abi {

// MSVC-mangled global operator new/delete symbols, /INCLUDE'd by the polycpp driver so
// polyreflect-rt's interposers win over vcruntime's. Suffix tags: Aligned = std::align_val_t,
// Nothrow = std::nothrow_t, Sized = size_t-sized delete.
inline constexpr char OperatorNew[] = "??2@YAPEAX_K@Z";
inline constexpr char OperatorNewAligned[] = "??2@YAPEAX_KW4align_val_t@std@@@Z";
inline constexpr char OperatorNewNothrow[] = "??2@YAPEAX_KAEBUnothrow_t@std@@@Z";
inline constexpr char OperatorNewAlignedNothrow[] = "??2@YAPEAX_KW4align_val_t@std@@AEBUnothrow_t@1@@Z";

inline constexpr char OperatorNewArray[] = "??_U@YAPEAX_K@Z";
inline constexpr char OperatorNewArrayAligned[] = "??_U@YAPEAX_KW4align_val_t@std@@@Z";
inline constexpr char OperatorNewArrayNothrow[] = "??_U@YAPEAX_KAEBUnothrow_t@std@@@Z";
inline constexpr char OperatorNewArrayAlignedNothrow[] = "??_U@YAPEAX_KW4align_val_t@std@@AEBUnothrow_t@1@@Z";

inline constexpr char OperatorDelete[] = "??3@YAXPEAX@Z";
inline constexpr char OperatorDeleteAligned[] = "??3@YAXPEAXW4align_val_t@std@@@Z";
inline constexpr char OperatorDeleteSized[] = "??3@YAXPEAX_K@Z";
inline constexpr char OperatorDeleteSizedAligned[] = "??3@YAXPEAX_KW4align_val_t@std@@@Z";
inline constexpr char OperatorDeleteNothrow[] = "??3@YAXPEAXAEBUnothrow_t@std@@@Z";
inline constexpr char OperatorDeleteAlignedNothrow[] = "??3@YAXPEAXW4align_val_t@std@@AEBUnothrow_t@1@@Z";

inline constexpr char OperatorDeleteArray[] = "??_V@YAXPEAX@Z";
inline constexpr char OperatorDeleteArrayAligned[] = "??_V@YAXPEAXW4align_val_t@std@@@Z";
inline constexpr char OperatorDeleteArraySized[] = "??_V@YAXPEAX_K@Z";
inline constexpr char OperatorDeleteArraySizedAligned[] = "??_V@YAXPEAX_KW4align_val_t@std@@@Z";
inline constexpr char OperatorDeleteArrayNothrow[] = "??_V@YAXPEAXAEBUnothrow_t@std@@@Z";
inline constexpr char OperatorDeleteArrayAlignedNothrow[] = "??_V@YAXPEAXW4align_val_t@std@@AEBUnothrow_t@1@@Z";

} // namespace polyregion::msvc_abi
