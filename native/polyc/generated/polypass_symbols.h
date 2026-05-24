// AUTO-GENERATED from PolyAST.PolyPassAbi via polyregion.ast.CodeGen. DO NOT EDIT.
#pragma once

#include "polyregion/polypass.h"

namespace polyregion::polypass::abi {

inline constexpr auto EnvPlugins = "POLYPASS_PLUGINS";

// ABI version the plugin was built against; polyc refuses mismatched plugins.
inline constexpr auto AbiVersion = "polypass_abi_version";

// Number of passes this plugin contributes.
inline constexpr auto PassCount = "polypass_pass_count";

// Bare identifier of the i-th pass (e.g. "FullOpt"). Process-lifetime; NULL if i out of range.
inline constexpr auto PassName = "polypass_pass_name";

// Optional human-readable description of the i-th pass; may return NULL or "".
inline constexpr auto PassDescr = "polypass_pass_descr";

// Run the NULL-terminated `steps` list against `in` (msgpack Program); steps share in-process state. On Ok, *out is a malloc'd
// PassRunResult; caller frees via polypass_free.
inline constexpr auto RunPasses = "polypass_run_passes";

// NUL-terminated diagnostic for the most recent non-Ok status. Valid until the next polypass_run_passes call; NULL when no error is set.
inline constexpr auto LastError = "polypass_last_error";

// Release a buffer returned by polypass_run_passes.
inline constexpr auto Free = "polypass_free";

using AbiVersionFn = polypass_abi_version_fn;
using PassCountFn = polypass_pass_count_fn;
using PassNameFn = polypass_pass_name_fn;
using PassDescrFn = polypass_pass_descr_fn;
using RunPassesFn = polypass_run_passes_fn;
using LastErrorFn = polypass_last_error_fn;
using FreeFn = polypass_free_fn;

} // namespace polyregion::polypass::abi
