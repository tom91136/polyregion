#pragma once

#include <cassert>
#include <cstdio>

#include "polyregion/types.h"
#include "ftypes.h"

#define POLYDCO_LOG(fmt, ...) std::fprintf(stderr, "[PolyDCO] " fmt "\n", __VA_ARGS__)

extern "C" {

[[maybe_unused]] POLYREGION_EXPORT void polydco_record(void *ptr, size_t size);
[[maybe_unused]] POLYREGION_EXPORT void polydco_release(void *ptr);
[[maybe_unused]] POLYREGION_EXPORT void polydco_debug_typelayout(const polyregion::runtime::TypeLayout *layout);

[[maybe_unused]] POLYREGION_EXPORT bool polydco_is_platformkind(polyregion::runtime::PlatformKind kind);
[[maybe_unused]] POLYREGION_EXPORT bool polydco_dispatch(int64_t lowerBoundInclusive, int64_t upperBoundInclusive, int64_t step,    //
                                                         polyregion::runtime::PlatformKind kind,                                    //
                                                         const polyregion::runtime::KernelBundle *bundle,                           //
                                                         size_t reductionsCount, const polyregion::polydco::FReduction *reductions, //
                                                         char *captures);
}