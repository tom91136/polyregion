#pragma once

#include <cassert>
#include <iostream>

#include "polyinvoke/runtime.h"

// XXX polystl-static bundles polyinvoke + LLVM Support; downstream linking goes through
// polycpp at runtime so the Windows system imports must be auto-linked from every TU.
#if defined(_MSC_VER)
  #pragma comment(lib, "Version.lib")
  #pragma comment(lib, "psapi.lib")
  #pragma comment(lib, "ntdll.lib")
  #pragma comment(lib, "ws2_32.lib")
  #pragma comment(lib, "ole32.lib")
  #pragma comment(lib, "shell32.lib")
  #pragma comment(lib, "advapi32.lib")
  #pragma comment(lib, "uuid.lib")
  #pragma comment(lib, "delayimp.lib")
#endif

template <polyregion::runtime::PlatformKind Kind, typename F>
POLYREGION_RT_PROTECT const polyregion::runtime::KernelBundle &__polyregion_offload__([[maybe_unused]] F) { // NOLINT(*-reserved-identifier)
  // Default body: returns an empty bundle so callers (e.g. polystl::details::parallel_for)
  // skip kernel dispatch and fall through to the host path. The polycpp clang plugin replaces
  // this body in offload-enabled builds (POLYCPP_NO_REWRITE not set).
  static const polyregion::runtime::KernelBundle empty{"", 0, nullptr, 0, nullptr, 0, ""};
  return empty;
}
namespace polyregion::polystl::details {

POLYREGION_RT_PROTECT POLYREGION_EXPORT void dispatchHostThreaded(size_t global, void *functorData, const char *moduleId);

POLYREGION_RT_PROTECT POLYREGION_EXPORT void dispatchManaged(size_t global, size_t local, size_t localMemBytes,
                                                             const runtime::TypeLayout *layout, void *functorData, const char *moduleId);
} // namespace polyregion::polystl::details