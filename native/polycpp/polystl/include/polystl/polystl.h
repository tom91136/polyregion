#pragma once

#include <cassert>
#include <iostream>

#include "polyinvoke/runtime.h"

template <polyregion::runtime::PlatformKind Kind, typename F>
POLYREGION_RT_PROTECT const polyregion::runtime::KernelBundle &__polyregion_offload__([[maybe_unused]] F) { // NOLINT(*-reserved-identifier)
  assert(false && "impl not replaced");
  std::abort();
}
namespace polyregion::polystl::details {

POLYREGION_RT_PROTECT POLYREGION_EXPORT void dispatchHostThreaded(size_t global, void *functorData, const char *moduleId);

POLYREGION_RT_PROTECT POLYREGION_EXPORT void dispatchManaged(size_t global, size_t local, size_t localMemBytes,
                                                             const runtime::TypeLayout *layout, void *functorData, const char *moduleId);
} // namespace polyregion::polystl::details