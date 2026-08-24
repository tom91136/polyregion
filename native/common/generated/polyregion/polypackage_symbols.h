// AUTO-GENERATED from PolyAST.PolyPackageAbi via polyregion.ast.CodeGen. DO NOT EDIT.
#pragma once

#include "polyregion/polypackage.h"

namespace polyregion::polypackage::abi {

// ABI version of the non-composable package service.
inline constexpr auto AbiVersion = "polypackage_abi_version";

// Link a versioned PackageLinkRequest into a Package encoded with the package-service wire schema.
inline constexpr auto LinkPackage = "polypackage_link_package";

// Resolve a versioned PackageSymRequest into a versioned PackageSymResolvedProgram.
inline constexpr auto ResolveSym = "polypackage_resolve_sym";

// NUL-terminated diagnostic for the most recent failed package operation; NULL when no error is set.
inline constexpr auto LastError = "polypackage_last_error";

// Release a buffer returned by a package operation.
inline constexpr auto Free = "polypackage_free";

using AbiVersionFn = polypackage_abi_version_fn;
using LinkPackageFn = polypackage_link_package_fn;
using ResolveSymFn = polypackage_resolve_sym_fn;
using LastErrorFn = polypackage_last_error_fn;
using FreeFn = polypackage_free_fn;

} // namespace polyregion::polypackage::abi
