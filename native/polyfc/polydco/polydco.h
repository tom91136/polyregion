#pragma once

#include <cassert>
#include <cstdio>

#include "polyinvoke/runtime.h"

#define POLYDCO_LOG(fmt, ...) std::fprintf(stderr, "[PolyDCO] " fmt "\n", __VA_ARGS__)

extern "C" {

struct FDim {
  uint64_t lowerBound, extent, stride;
};

struct FArrayDesc {
  void *addr;
  uint64_t sizeInBytes;
  uint64_t ranks;
  FDim *dims;
};

[[maybe_unused]] POLYREGION_EXPORT void polydco_record(void *ptr, size_t size);
[[maybe_unused]] POLYREGION_EXPORT void polydco_release(void *ptr);
[[maybe_unused]] POLYREGION_EXPORT void polydco_debug_farraydesc(FArrayDesc *desc);
[[maybe_unused]] POLYREGION_EXPORT void polydco_debug_typelayout(polyregion::runtime::TypeLayout *layout);

[[maybe_unused]] POLYREGION_EXPORT bool polydco_is_platformkind(polyregion::runtime::PlatformKind kind);
[[maybe_unused]] POLYREGION_EXPORT bool polydco_dispatch(int64_t lowerBoundInclusive, int64_t upperBoundInclusive, int64_t step, //
                                                         polyregion::runtime::PlatformKind kind,                                 //
                                                         const polyregion::runtime::KernelBundle *bundle,                        //
                                                         char *captures);
}