#pragma once

#include <algorithm>
#include <cassert>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "polyregion/concurrency_utils.hpp"
#include "polyregion/conventions.h"
#include "polyrt/rt.h"

#include "polystl.h"

[[nodiscard]] uint32_t __polyregion_builtin_gpu_global_idx(uint32_t);  // NOLINT(*-reserved-identifier)
[[nodiscard]] uint32_t __polyregion_builtin_gpu_global_size(uint32_t); // NOLINT(*-reserved-identifier)

[[nodiscard]] uint32_t __polyregion_builtin_gpu_group_idx(uint32_t);  // NOLINT(*-reserved-identifier)
[[nodiscard]] uint32_t __polyregion_builtin_gpu_group_size(uint32_t); // NOLINT(*-reserved-identifier)

[[nodiscard]] uint32_t __polyregion_builtin_gpu_local_idx(uint32_t);  // NOLINT(*-reserved-identifier)
[[nodiscard]] uint32_t __polyregion_builtin_gpu_local_size(uint32_t); // NOLINT(*-reserved-identifier)

void __polyregion_builtin_gpu_barrier_global(); // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_barrier_local();  // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_barrier_all();    // NOLINT(*-reserved-identifier)

void __polyregion_builtin_gpu_fence_global(); // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_fence_local();  // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_fence_all();    // NOLINT(*-reserved-identifier)

extern "C" void __polyregion_builtin_assert(uint32_t code, const char *message); // NOLINT(*-reserved-identifier)

namespace polyregion::polystl::details {

using polyrt::DebugLevel;

// XXX passthrough builds (POLYCPP_NO_REWRITE) leave moduleName empty; host-fallback in that case.
inline bool bundleIsRewritten(const polyrt::KernelBundle &bundle) { return bundle.moduleName != nullptr && bundle.moduleName[0] != '\0'; }

// XXX struct-member access emits llvm.ptr.annotation on the loaded value; a local-var annotate
// would emit llvm.var.annotation on the alloca and detach after mem2reg.
// always_inline so the annotation lands in the caller even at -O0; otherwise the marked
// pointer is only visible inside this helper and the polyreflect pass can't taint the caller.
struct __PolyreflectTrackPtr {
  [[clang::annotate(POLYREFLECT_TRACK_ANNOTATION)]] void *p;
};
[[gnu::always_inline]] inline void *polyreflectTrackPtr(void *p) { return __PolyreflectTrackPtr{p}.p; }

template <class UnaryFunction> void parallel_for(int64_t global, UnaryFunction f) {
  polyrt::initialise();
  auto N = std::thread::hardware_concurrency();
  log(DebugLevel::Debug, "<%s, %ld> Dispatch", __func__, global);

  if (!polyrt::currentPlatform) {
    if (!polyrt::hostFallback()) {
      polyrt::noCompatibleKernelExit("parallel_for");
      return;
    }
    for (int64_t i = 0; i < global; ++i)
      f(i);
    return;
  }

  switch (polyrt::currentPlatform->kind()) {
    case polyregion::runtime::PlatformKind::HostThreaded: {
      auto [b, e] = concurrency_utils::splitStaticExclusive<int64_t>(0, global, N);
      const int64_t *begin = b.data();
      const int64_t *end = e.data();
      auto kernel = [&f, begin, end](const int64_t tid) {
        for (int64_t i = begin[tid]; i < end[tid]; ++i) {
          f(i);
        }
      };

      const polyrt::KernelBundle &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::HostThreaded>(kernel);
      if (!bundleIsRewritten(bundle)) {
        for (int64_t i = 0; i < global; ++i)
          f(i);
        return;
      }
      for (size_t i = 0; i < bundle.objectCount; ++i) {
        std::string loadedModule;
        if (!polyrt::loadKernelObject(bundle.moduleName, bundle.objects[i], &kernel,
                                      bundle.structCount ? &bundle.structs[bundle.interfaceLayoutIdx] : nullptr, &loadedModule))
          continue;
        details::dispatchHostThreaded(b.size(), polyreflectTrackPtr(&kernel), loadedModule.c_str(), bundle.asserts);
        return;
      }
      break;
    }
    case polyregion::runtime::PlatformKind::Managed: {
      auto kernel = [f, global]() {
        const auto lim = static_cast<uint32_t>(global);
        const auto gid = __polyregion_builtin_gpu_global_idx(0);
        const auto gs = __polyregion_builtin_gpu_global_size(0);
        for (uint32_t i = gid; i < lim; i += gs) {
          f(static_cast<int64_t>(i));
        }
      };
      int64_t blocks = program_meta::VkWorkgroupSizeXValue;

      const polyrt::KernelBundle &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::Managed>(kernel);
      if (!bundleIsRewritten(bundle)) {
        for (int64_t i = 0; i < global; ++i)
          f(i);
        return;
      }

      if (polyrt::debugLevel() >= DebugLevel::Debug) {
        for (size_t i = 0; i < bundle.structCount; ++i) {
          if (i == bundle.interfaceLayoutIdx) fprintf(stderr, "**Exported**\n");
          bundle.structs[i].visualise(stderr);
        }
      }

      void *kernelPtr = polyreflectTrackPtr(&kernel);
      for (size_t i = 0; i < bundle.objectCount; ++i) {
        std::string loadedModule;
        if (!polyrt::loadKernelObject(bundle.moduleName, bundle.objects[i], &kernel,
                                      bundle.structCount ? &bundle.structs[bundle.interfaceLayoutIdx] : nullptr, &loadedModule))
          continue;
        const auto grid = (global + blocks - 1) / blocks;
        details::dispatchManaged(grid, blocks, 0, &bundle.structs[bundle.interfaceLayoutIdx], kernelPtr, loadedModule.c_str(),
                                 bundle.prelude, bundle.postlude, bundle.asserts);
        return;
      }
      break;
    }
  }

  if (!polyrt::hostFallback()) {
    polyrt::noCompatibleKernelExit("parallel_for");
    return;
  }
  log(DebugLevel::Debug, "<%s, %d> Host fallback", __func__, global);
  for (int64_t i = 0; i < global; ++i) {
    f(i);
  }
}

template <typename T, class UnaryFunction, class BinaryFunction>
T parallel_reduce(int64_t global, T init, UnaryFunction f, BinaryFunction reduce) {
  polyrt::initialise();
  auto N = std::thread::hardware_concurrency();
  log(DebugLevel::Debug, "<%s, %d> Dispatch", __func__, global);

  if (!polyrt::currentPlatform) {
    if (!polyrt::hostFallback()) {
      polyrt::noCompatibleKernelExit("parallel_reduce");
      return init;
    }
    T acc = init;
    for (int64_t i = 0; i < global; ++i)
      acc = reduce(acc, f(i));
    return acc;
  }

  switch (polyrt::currentPlatform->kind()) {
    case polyregion::runtime::PlatformKind ::HostThreaded: {
      auto [b, e] = concurrency_utils::splitStaticExclusive<int64_t>(0, global, N);
      const int64_t groups = b.size();
      const int64_t *begin = b.data();
      const int64_t *end = e.data();

      std::vector<T> groupPartial(groups);
      auto kernel = [&f, &reduce, init, begin, end, out = groupPartial.data()](const int64_t tid) {
        auto acc = init;
        for (int64_t i = begin[tid]; i < end[tid]; ++i) {
          acc = reduce(acc, f(i));
        }
        out[tid] = acc;
      };
      const polyrt::KernelBundle &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::HostThreaded>(kernel);
      if (!bundleIsRewritten(bundle)) {
        T acc = init;
        for (int64_t i = 0; i < global; ++i)
          acc = reduce(acc, f(i));
        return acc;
      }
      for (size_t i = 0; i < bundle.objectCount; ++i) {
        std::string loadedModule;
        if (!polyrt::loadKernelObject(bundle.moduleName, bundle.objects[i], &kernel,
                                      bundle.structCount ? &bundle.structs[bundle.interfaceLayoutIdx] : nullptr, &loadedModule))
          continue;
        details::dispatchHostThreaded(groups, polyreflectTrackPtr(&kernel), loadedModule.c_str(), bundle.asserts);
        T acc = init;
        for (int64_t groupIdx = 0; groupIdx < groups; ++groupIdx) {
          acc = reduce(acc, groupPartial[groupIdx]);
        }
        return acc;
      }
      break;
    }
    case polyregion::runtime::PlatformKind::Managed: {
      const int64_t groups = program_meta::VkWorkgroupSizeXValue;
      std::vector<T> groupPartial(groups);
      auto kernel = [out = groupPartial.data(), init, f, reduce,
                     global]([[clang::annotate(POLYREGION_LOCAL_ANNOTATION)]] T *localPartialSum) {
        const auto lim = static_cast<uint32_t>(global);
        const auto localIdx = __polyregion_builtin_gpu_local_idx(0);
        localPartialSum[localIdx] = init;
        const auto gid = __polyregion_builtin_gpu_global_idx(0);
        const auto gs = __polyregion_builtin_gpu_global_size(0);
        for (uint32_t i = gid; i < lim; i += gs) {
          localPartialSum[localIdx] = reduce(localPartialSum[localIdx], f(static_cast<int64_t>(i)));
        }
        for (uint32_t offset = __polyregion_builtin_gpu_local_size(0) / 2; offset > 0; offset /= 2) {
          __polyregion_builtin_gpu_barrier_local();
          if (localIdx < offset) {
            localPartialSum[localIdx] = reduce(localPartialSum[localIdx], localPartialSum[localIdx + offset]);
          }
        }
        if (localIdx == 0) {
          out[__polyregion_builtin_gpu_group_idx(0)] = localPartialSum[localIdx];
        }
      };
      const polyrt::KernelBundle &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::Managed>(kernel);
      if (!bundleIsRewritten(bundle)) {
        T acc = init;
        for (int64_t i = 0; i < global; ++i)
          acc = reduce(acc, f(i));
        return acc;
      }
      void *kernelPtr = polyreflectTrackPtr(&kernel);
      for (size_t i = 0; i < bundle.objectCount; ++i) {
        std::string loadedModule;
        if (!polyrt::loadKernelObject(bundle.moduleName, bundle.objects[i], &kernel,
                                      bundle.structCount ? &bundle.structs[bundle.interfaceLayoutIdx] : nullptr, &loadedModule))
          continue;
        details::dispatchManaged(program_meta::VkWorkgroupSizeXValue, groups, groups * sizeof(T),
                                 &bundle.structs[bundle.interfaceLayoutIdx], kernelPtr, loadedModule.c_str(), bundle.prelude,
                                 bundle.postlude, bundle.asserts);
        T acc = init;
        for (int64_t groupIdx = 0; groupIdx < groups; ++groupIdx) {
          acc = reduce(acc, groupPartial[groupIdx]);
        }
        return acc;
      }
      break;
    }
  }

  if (!polyrt::hostFallback()) {
    polyrt::noCompatibleKernelExit("parallel_reduce");
    return init;
  }
  log(DebugLevel::Debug, "<%s, %ld> Host fallback", __func__, global);

  T acc = init;
  for (int64_t globalIdx = 0; globalIdx < global; ++globalIdx) {
    acc = reduce(acc, f(globalIdx));
  }
  return acc;
}

} // namespace polyregion::polystl::details

namespace std { // NOLINT(*-dcl58-cpp)

// XXX Policy types are always defined here; is_execution_policy is taken from the stdlib when
// available, polyfilled otherwise.
namespace execution {
class sequenced_policy {};
class parallel_policy {};
class parallel_unsequenced_policy {};
class unsequenced_policy {};

[[maybe_unused]] inline constexpr sequenced_policy seq{};
[[maybe_unused]] inline constexpr parallel_policy par{};
[[maybe_unused]] inline constexpr parallel_unsequenced_policy par_unseq{};
[[maybe_unused]] inline constexpr unsequenced_policy unseq{};
} // namespace execution

#if !(defined(_HAS_CXX17) && _HAS_CXX17) && !defined(__cpp_lib_execution)
template <typename> struct is_execution_policy : std::false_type {};
template <> struct is_execution_policy<execution::sequenced_policy> : std::true_type {};
template <> struct is_execution_policy<execution::parallel_policy> : std::true_type {};
template <> struct is_execution_policy<execution::parallel_unsequenced_policy> : std::true_type {};
template <> struct is_execution_policy<execution::unsequenced_policy> : std::true_type {};
template <typename T> constexpr bool is_execution_policy_v = is_execution_policy<T>::value;
#endif

// XXX par_unseq overloads use the concrete policy type so partial ordering routes them here;
// other policies defer to the stdlib or the polyfill below.

template <class ForwardIt, class UnaryFunction>
void for_each(execution::parallel_unsequenced_policy, ForwardIt first, ForwardIt last, UnaryFunction f) {
  polyregion::polystl::details::parallel_for(last - first, [f, first](auto idx) { f(*(first + idx)); });
}

template <class ForwardIt, class T> void fill(execution::parallel_unsequenced_policy, ForwardIt first, ForwardIt last, const T &value) {
  // XXX `value = value` strips the T& so the capture is by value.
  polyregion::polystl::details::parallel_for(last - first, [value = value, first](auto idx) { (*(first + idx)) = value; });
}

template <class ForwardIt, class Size, class UnaryFunction>
ForwardIt for_each_n(execution::parallel_unsequenced_policy, ForwardIt first, Size n, UnaryFunction f) {
  polyregion::polystl::details::parallel_for(static_cast<int64_t>(n), [f, first](auto idx) { f(*(first + idx)); });
  return first + n;
}

template <class ForwardIt, class Size, class T>
ForwardIt fill_n(execution::parallel_unsequenced_policy, ForwardIt first, Size n, const T &value) {
  // XXX `value = value` strips the T& so the capture is by value.
  polyregion::polystl::details::parallel_for(static_cast<int64_t>(n), [value = value, first](auto idx) { (*(first + idx)) = value; });
  return first + n;
}

template <class ForwardIt1, class ForwardIt2>
ForwardIt2 copy(execution::parallel_unsequenced_policy, ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first) {
  polyregion::polystl::details::parallel_for(last - first, [d_first, first](auto idx) { (*(d_first + idx)) = (*(first + idx)); });
  return d_first;
}

template <class ForwardIt1, class ForwardIt2, class UnaryOperation>
ForwardIt2 transform(execution::parallel_unsequenced_policy, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 d_first,
                     UnaryOperation unary_op) {
  polyregion::polystl::details::parallel_for(last1 - first1,
                                             [d_first, first1, unary_op](auto idx) { *(d_first + idx) = unary_op(*(first1 + idx)); });
  return d_first;
}

template <class ForwardIt1, class ForwardIt2, class ForwardIt3, class BinaryOperation>
ForwardIt3 transform(execution::parallel_unsequenced_policy, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt3 d_first,
                     BinaryOperation binary_op) {
  polyregion::polystl::details::parallel_for(
      last1 - first1, [d_first, first1, first2, binary_op](auto idx) { (*(d_first + idx)) = binary_op(*(first1 + idx), *(first2 + idx)); });
  return d_first;
}

template <class ForwardIt1, class ForwardIt2, class T>
T transform_reduce(execution::parallel_unsequenced_policy, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init) {
  return polyregion::polystl::details::parallel_reduce(
      last1 - first1, init, //
      [first1, first2](auto idx) { return *(first1 + idx) * *(first2 + idx); }, [](auto l, auto r) { return l + r; });
}

template <class ForwardIt1, class ForwardIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
T transform_reduce(execution::parallel_unsequenced_policy, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init,
                   BinaryReductionOp reduce, BinaryTransformOp transform) {
  // XXX Wrap `reduce` in a by-value lambda; perfect-forwarded references confuse SPIR-V
  // storage-class tracking and IGC emits null pointers.
  return polyregion::polystl::details::parallel_reduce(
      last1 - first1, init, [transform, first1, first2](auto idx) { return transform(*(first1 + idx), *(first2 + idx)); },
      [reduce](auto l, auto r) { return reduce(l, r); });
}

template <class ForwardIt, class T, class BinaryReductionOp, class UnaryTransformOp>
T transform_reduce(execution::parallel_unsequenced_policy, ForwardIt first, ForwardIt last, T init, BinaryReductionOp reduce,
                   UnaryTransformOp transform) {
  return polyregion::polystl::details::parallel_reduce(last - first, init, transform, [reduce](auto l, auto r) { return reduce(l, r); });
}

// Sequential fallbacks for par/seq/unseq when no stdlib <execution> is available.
#if !(defined(_HAS_CXX17) && _HAS_CXX17) && !defined(__cpp_lib_execution)
template <class ExecutionPolicy, class ForwardIt, class UnaryFunction>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, void> //
for_each(ExecutionPolicy &&, ForwardIt first, ForwardIt last, UnaryFunction f) {
  std::for_each(first, last, f);
}

template <class ExecutionPolicy, class ForwardIt, class T>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, void> //
fill(ExecutionPolicy &&, ForwardIt first, ForwardIt last, const T &value) {
  std::fill(first, last, value);
}

template <class ExecutionPolicy, class ForwardIt, class Size, class UnaryFunction>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, ForwardIt> //
for_each_n(ExecutionPolicy &&, ForwardIt first, Size n, UnaryFunction f) {
  return std::for_each_n(first, n, f);
}

template <class ExecutionPolicy, class ForwardIt, class Size, class T>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, ForwardIt> //
fill_n(ExecutionPolicy &&, ForwardIt first, Size n, const T &value) {
  return std::fill_n(first, n, value);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, ForwardIt2> //
copy(ExecutionPolicy &&, ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first) {
  return std::copy(first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class UnaryOperation>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, ForwardIt2> //
transform(ExecutionPolicy &&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 d_first, UnaryOperation unary_op) {
  return std::transform(first1, last1, d_first, unary_op);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class ForwardIt3, class BinaryOperation>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, ForwardIt3> //
transform(ExecutionPolicy &&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt3 d_first, BinaryOperation binary_op) {
  return std::transform(first1, last1, first2, d_first, binary_op);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, T> //
transform_reduce(ExecutionPolicy &&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init) {
  return std::transform_reduce(first1, last1, first2, init);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, T> //
transform_reduce(ExecutionPolicy &&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init, BinaryReductionOp reduce,
                 BinaryTransformOp transform) {
  return std::transform_reduce(first1, last1, first2, init, reduce, transform);
}

template <class ExecutionPolicy, class ForwardIt, class T, class BinaryReductionOp, class UnaryTransformOp>
std::enable_if_t<is_execution_policy_v<std::decay_t<ExecutionPolicy>>, T> //
transform_reduce(ExecutionPolicy &&, ForwardIt first, ForwardIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform) {
  return std::transform_reduce(first, last, init, reduce, transform);
}
#endif

} // namespace std
