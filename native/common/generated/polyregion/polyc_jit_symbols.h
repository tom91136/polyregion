// AUTO-GENERATED from PolyAST.PolyJitAbi via polyregion.ast.CodeGen. DO NOT EDIT.
#pragma once

#include "polyregion/polyc_jit.h"

namespace polyregion::polyc_jit::abi {

// Compile a msgpack Program. Free the result with polyc_jit_free.
inline constexpr auto Compile = "polyc_jit_compile";

// NUL-terminated diagnostic for the most recent non-Ok status; valid until the next polyc_jit_compile call, NULL when none.
inline constexpr auto LastError = "polyc_jit_last_error";

// Release a buffer returned by polyc_jit_compile.
inline constexpr auto Free = "polyc_jit_free";

using CompileFn = polyc_jit_compile_fn;
using LastErrorFn = polyc_jit_last_error_fn;
using FreeFn = polyc_jit_free_fn;

} // namespace polyregion::polyc_jit::abi
