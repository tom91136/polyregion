#pragma once

#include <cassert>
#include <iostream>

#include "polyinvoke/runtime.h"

#define POLYSTL_LOG(fmt, ...) std::fprintf(stderr, "[PolySTL] " fmt "\n", __VA_ARGS__)

template <polyregion::runtime::PlatformKind Kind, typename F>
const polyregion::runtime::KernelBundle &__polyregion_offload__([[maybe_unused]] F) { // NOLINT(*-reserved-identifier)
  assert(false && "impl not replaced");
  std::abort();
}
namespace polyregion::polystl::details {
void initialise();
void dispatchManaged(size_t global, size_t local, size_t localMemBytes, const runtime::TypeLayout *layout, void *functorData,
                     const char *moduleId);
} // namespace polyregion::polystl::details