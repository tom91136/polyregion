#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "polyregion/polyc_jit.h"

namespace polyregion::jit {

std::string compilerIdentity();

std::string cachePath(const uint8_t *program, size_t programLen, uint8_t target, const char *arch, const char *pipelineSpec, uint8_t opt,
                      const polyc_jit_spec_const_t *specs, size_t nSpecs, const std::string &identity);

} // namespace polyregion::jit
