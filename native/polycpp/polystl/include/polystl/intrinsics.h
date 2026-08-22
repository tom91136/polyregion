#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>

// C-linkage remapper hooks. Their inline bodies define the single-work-item host fallback;
// PolyCPP replaces calls inside an offloaded region with the corresponding target SpecOp.
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_global_idx(uint32_t) { return 0; }  // NOLINT(*-reserved-identifier)
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_global_size(uint32_t) { return 1; } // NOLINT(*-reserved-identifier)
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_group_idx(uint32_t) { return 0; }   // NOLINT(*-reserved-identifier)
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_group_size(uint32_t) { return 1; }  // NOLINT(*-reserved-identifier)
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_local_idx(uint32_t) { return 0; }   // NOLINT(*-reserved-identifier)
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_local_size(uint32_t) { return 1; }  // NOLINT(*-reserved-identifier)
extern "C" inline void __polyregion_gpu_barrier_global() {}                                   // NOLINT(*-reserved-identifier)
extern "C" inline void __polyregion_gpu_barrier_local() {}                                    // NOLINT(*-reserved-identifier)
extern "C" inline void __polyregion_gpu_barrier_all() {}                                      // NOLINT(*-reserved-identifier)
extern "C" inline void __polyregion_gpu_fence_global() {}                                     // NOLINT(*-reserved-identifier)
extern "C" inline void __polyregion_gpu_fence_local() {}                                      // NOLINT(*-reserved-identifier)
extern "C" inline void __polyregion_gpu_fence_all() {}                                        // NOLINT(*-reserved-identifier)
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_lane_idx() { return 0; }            // NOLINT(*-reserved-identifier)
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_subgroup_size() { return 1; }       // NOLINT(*-reserved-identifier)

// `clamp` is an inclusive power-of-two segment mask. UINT32_MAX resolves to the active mask on
// targets that require one explicitly. Native subgroup operations require subgroup-uniform
// participation; emulated targets currently require whole-workgroup-uniform participation.
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_shuffle_down_u32( // NOLINT(*-reserved-identifier)
    uint32_t value, [[maybe_unused]] uint32_t delta, [[maybe_unused]] uint32_t clamp, [[maybe_unused]] uint32_t mask = UINT32_MAX) {
  return value;
}
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_shuffle_up_u32( // NOLINT(*-reserved-identifier)
    uint32_t value, [[maybe_unused]] uint32_t delta, [[maybe_unused]] uint32_t clamp, [[maybe_unused]] uint32_t mask = UINT32_MAX) {
  return value;
}
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_shuffle_idx_u32( // NOLINT(*-reserved-identifier)
    uint32_t value, [[maybe_unused]] uint32_t sourceLane, [[maybe_unused]] uint32_t clamp, [[maybe_unused]] uint32_t mask = UINT32_MAX) {
  return value;
}
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_shuffle_xor_u32( // NOLINT(*-reserved-identifier)
    uint32_t value, [[maybe_unused]] uint32_t laneMask, [[maybe_unused]] uint32_t clamp, [[maybe_unused]] uint32_t mask = UINT32_MAX) {
  return value;
}
extern "C" inline void __polyregion_gpu_subgroup_barrier([[maybe_unused]] uint32_t mask = UINT32_MAX) {} // NOLINT(*-reserved-identifier)
extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_ballot(                                        // NOLINT(*-reserved-identifier)
    uint32_t mask, bool predicate) {
  return predicate && (mask & 1u) != 0u ? 1u : 0u;
}
extern "C" [[nodiscard]] inline bool __polyregion_gpu_vote_any(uint32_t mask, bool predicate) { // NOLINT(*-reserved-identifier)
  return predicate && (mask & 1u) != 0u;
}
extern "C" [[nodiscard]] inline bool __polyregion_gpu_vote_all(uint32_t mask, bool predicate) { // NOLINT(*-reserved-identifier)
  return predicate || (mask & 1u) == 0u;
}

namespace polyregion::polystl::details {

#if !defined(__clang__) && !defined(__GNUC__)
inline std::atomic_flag atomicFallbackLock = ATOMIC_FLAG_INIT;

struct AtomicFallbackGuard {
  AtomicFallbackGuard() {
    while (atomicFallbackLock.test_and_set(std::memory_order_acquire)) {
    }
  }
  ~AtomicFallbackGuard() { atomicFallbackLock.clear(std::memory_order_release); }
};
#endif

inline uint32_t atomicMinMax(uint32_t *ptr, uint32_t value, bool minimum) {
#if defined(__clang__) || defined(__GNUC__)
  uint32_t previous;
  __atomic_load(ptr, &previous, __ATOMIC_RELAXED);
  for (;;) {
    uint32_t desired = minimum ? std::min(previous, value) : std::max(previous, value);
    if (__atomic_compare_exchange(ptr, &previous, &desired, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return previous;
  }
#else
  AtomicFallbackGuard guard;
  const auto previous = *ptr;
  *ptr = minimum ? std::min(previous, value) : std::max(previous, value);
  return previous;
#endif
}

} // namespace polyregion::polystl::details

// Atomic operations use device scope and relaxed ordering on an offload target.
extern "C" inline uint32_t __polyregion_gpu_atomic_xchg_u32(uint32_t *ptr, uint32_t value) { // NOLINT(*-reserved-identifier)
#if defined(__clang__) || defined(__GNUC__)
  return __atomic_exchange_n(ptr, value, __ATOMIC_RELAXED);
#else
  polyregion::polystl::details::AtomicFallbackGuard guard;
  const auto previous = *ptr;
  *ptr = value;
  return previous;
#endif
}
extern "C" inline uint32_t __polyregion_gpu_atomic_add_u32(uint32_t *ptr, uint32_t value) { // NOLINT(*-reserved-identifier)
#if defined(__clang__) || defined(__GNUC__)
  return __atomic_fetch_add(ptr, value, __ATOMIC_RELAXED);
#else
  polyregion::polystl::details::AtomicFallbackGuard guard;
  const auto previous = *ptr;
  *ptr += value;
  return previous;
#endif
}
extern "C" inline uint32_t __polyregion_gpu_atomic_sub_u32(uint32_t *ptr, uint32_t value) { // NOLINT(*-reserved-identifier)
#if defined(__clang__) || defined(__GNUC__)
  return __atomic_fetch_sub(ptr, value, __ATOMIC_RELAXED);
#else
  polyregion::polystl::details::AtomicFallbackGuard guard;
  const auto previous = *ptr;
  *ptr -= value;
  return previous;
#endif
}
extern "C" inline uint32_t __polyregion_gpu_atomic_min_u32(uint32_t *ptr, uint32_t value) { // NOLINT(*-reserved-identifier)
  return polyregion::polystl::details::atomicMinMax(ptr, value, true);
}
extern "C" inline uint32_t __polyregion_gpu_atomic_max_u32(uint32_t *ptr, uint32_t value) { // NOLINT(*-reserved-identifier)
  return polyregion::polystl::details::atomicMinMax(ptr, value, false);
}
extern "C" inline uint32_t __polyregion_gpu_atomic_and_u32(uint32_t *ptr, uint32_t value) { // NOLINT(*-reserved-identifier)
#if defined(__clang__) || defined(__GNUC__)
  return __atomic_fetch_and(ptr, value, __ATOMIC_RELAXED);
#else
  polyregion::polystl::details::AtomicFallbackGuard guard;
  const auto previous = *ptr;
  *ptr &= value;
  return previous;
#endif
}
extern "C" inline uint32_t __polyregion_gpu_atomic_or_u32(uint32_t *ptr, uint32_t value) { // NOLINT(*-reserved-identifier)
#if defined(__clang__) || defined(__GNUC__)
  return __atomic_fetch_or(ptr, value, __ATOMIC_RELAXED);
#else
  polyregion::polystl::details::AtomicFallbackGuard guard;
  const auto previous = *ptr;
  *ptr |= value;
  return previous;
#endif
}
extern "C" inline uint32_t __polyregion_gpu_atomic_xor_u32(uint32_t *ptr, uint32_t value) { // NOLINT(*-reserved-identifier)
#if defined(__clang__) || defined(__GNUC__)
  return __atomic_fetch_xor(ptr, value, __ATOMIC_RELAXED);
#else
  polyregion::polystl::details::AtomicFallbackGuard guard;
  const auto previous = *ptr;
  *ptr ^= value;
  return previous;
#endif
}

extern "C" [[nodiscard]] inline uint32_t __polyregion_gpu_volatile_load_u32(const uint32_t *ptr) { // NOLINT(*-reserved-identifier)
  return *reinterpret_cast<const volatile uint32_t *>(ptr);
}
extern "C" inline void __polyregion_gpu_volatile_store_u32(uint32_t *ptr, uint32_t value) { // NOLINT(*-reserved-identifier)
  *reinterpret_cast<volatile uint32_t *>(ptr) = value;
}

extern "C" void __polyregion_builtin_assert(uint32_t code, const char *message); // NOLINT(*-reserved-identifier)
