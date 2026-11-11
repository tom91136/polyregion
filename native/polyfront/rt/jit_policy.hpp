#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "aspartame/all.hpp"

namespace polyregion::polyrt {

using namespace aspartame;

struct JitPolicyChoice {
  bool specialise;
  bool admitted;
};

class AdaptiveJitPolicy {
  struct Observation {
    size_t hits;
    uint64_t touched;
  };
  struct KernelState {
    bool genericSeen = false;
    std::unordered_set<uint64_t> variants;
    std::unordered_map<uint64_t, Observation> observations;
  };

  size_t hotThreshold;
  size_t variantLimit;
  size_t observationLimit;
  uint64_t tick = 0;
  std::unordered_map<std::string, KernelState> kernels;

  void observe(KernelState &state, const uint64_t key) {
    if (const auto it = state.observations.find(key); it != state.observations.end()) {
      it->second.hits++;
      it->second.touched = ++tick;
      return;
    }
    if (state.observations.size() >= observationLimit) {
      state.observations                                                               //
          | map([](const auto &key, const auto &observation) {                         //
              return std::pair{key, std::pair{observation.hits, observation.touched}}; //
            })                                                                         //
          | min_by([](const auto &, const auto &rank) { return rank; })                //
          | for_each([&](const auto &victim, const auto &) { state.observations.erase(victim); });
    }
    state.observations.emplace(key, Observation{.hits = 1, .touched = ++tick});
  }

public:
  explicit AdaptiveJitPolicy(const size_t hotThreshold = 3, const size_t variantLimit = 8, const size_t observationLimit = 0)
      : hotThreshold(std::max<size_t>(1, hotThreshold)), variantLimit(variantLimit),
        observationLimit(observationLimit ? observationLimit : std::max<size_t>(16, variantLimit * 4)) {}

  JitPolicyChoice select(const std::string &kernel, const uint64_t key) {
    auto &state = kernels[kernel];
    if (!state.genericSeen) {
      state.genericSeen = true;
      if (variantLimit) observe(state, key);
      return {.specialise = false, .admitted = false};
    }
    if (state.variants.contains(key)) return {.specialise = true, .admitted = false};
    if (!variantLimit || state.variants.size() >= variantLimit) return {.specialise = false, .admitted = false};

    observe(state, key);
    const auto observation = state.observations.find(key);
    if (observation->second.hits < hotThreshold) return {.specialise = false, .admitted = false};
    state.observations.erase(observation);
    state.variants.insert(key);
    return {.specialise = true, .admitted = true};
  }

  size_t variantCount(const std::string &kernel) const {
    if (const auto it = kernels.find(kernel); it != kernels.end()) return it->second.variants.size();
    return 0;
  }
  size_t observationCount(const std::string &kernel) const {
    if (const auto it = kernels.find(kernel); it != kernels.end()) return it->second.observations.size();
    return 0;
  }
};

} // namespace polyregion::polyrt
