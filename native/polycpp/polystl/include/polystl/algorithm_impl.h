#pragma once

#include <algorithm>
#include <iostream>
#include <thread>
#include <type_traits>

#include "polystl.h"

// #include_next <execution>

namespace std::execution { // NOLINT(*-dcl58-cpp)
class sequenced_policy {};
class parallel_policy {};
class parallel_unsequenced_policy {};
class unsequenced_policy {};

[[maybe_unused]] inline constexpr sequenced_policy seq{};
[[maybe_unused]] inline constexpr parallel_policy par{};
[[maybe_unused]] inline constexpr parallel_unsequenced_policy par_unseq{};
[[maybe_unused]] inline constexpr unsequenced_policy unseq{};
} // namespace std::execution

namespace std {
template <typename> struct is_execution_policy : std::false_type {};                                     // NOLINT(*-dcl58-cpp)
template <> struct is_execution_policy<std::execution::sequenced_policy> : std::true_type {};            // NOLINT(*-dcl58-cpp)
template <> struct is_execution_policy<std::execution::parallel_policy> : std::true_type {};             // NOLINT(*-dcl58-cpp)
template <> struct is_execution_policy<std::execution::parallel_unsequenced_policy> : std::true_type {}; // NOLINT(*-dcl58-cpp)
template <> struct is_execution_policy<std::execution::unsequenced_policy> : std::true_type {};          // NOLINT(*-dcl58-cpp)
template <typename T> constexpr bool is_execution_policy_v = is_execution_policy<T>::value;              // NOLINT(*-dcl58-cpp)
} // namespace std

namespace std {

namespace {

template <class UnaryFunction> void parallel_for(int64_t global, UnaryFunction f) {
  auto N = std::thread::hardware_concurrency();
  POLYSTL_LOG("<%s, %d> Dispatch", __func__, global);

  if (PlatformKind kind; __polyregion_platform_kind(kind)) {
    switch (kind) {
      case polyregion::runtime::PlatformKind ::HostThreaded: {
        auto [b, e] = polyregion::concurrency_utils::splitStaticExclusive2<int64_t>(0, global, 1);
        const int64_t *begin = b.data();
        const int64_t *end = e.data();
        const auto kernel = [&f, begin, end](const int64_t tid) {
          for (int64_t i = begin[tid]; i < end[tid]; ++i) {
            f(i);
          }
        };
        auto &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::HostThreaded>(kernel);
        std::byte argData[sizeof(decltype(kernel))];
        std::memcpy(argData, &kernel, sizeof(decltype(kernel)));
        for (size_t i = 0; i < bundle.objectCount; ++i) {
          if (__polyregion_dispatch_hostthreaded(b.size(), &argData, bundle.moduleName, bundle.get(i))) return;
        }
        break;
      }
      case polyregion::runtime::PlatformKind ::Managed: {
        const auto kernel = [f, global]() {
          const auto lim = static_cast<uint32_t>(global);
          const auto gid = __polyregion_builtin_gpu_global_idx(0);
          const auto gs = __polyregion_builtin_gpu_global_size(0);
          for (uint32_t i = gid; i < lim; i += gs) {
            f(static_cast<int64_t>(i));
          }
        };
        int64_t blocks = 256;
        auto &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::Managed>(kernel);
        for (size_t i = 0; i < bundle.objectCount; ++i) {
          if (__polyregion_dispatch_managed(global / blocks, blocks, 0, sizeof(decltype(kernel)), &kernel, bundle.moduleName,
                                            bundle.get(i)))
            return;
        }
        break;
      }
    }
  }

  static constexpr const char *HostFallbackEnv = "POLYSTL_HOST_FALLBACK";

  if (auto env = std::getenv(HostFallbackEnv); env) {
    errno = 0; // strtol to avoid exceptions
    size_t value = std::strtol(env, nullptr, 10);
    if (errno == 0 && value == 0) {
      POLYSTL_LOG("<%s, %d> No compatible backend and host fallback disabled, returning...", global, __func__);
      return;
    }
  } // default is to use host fallback, so keep going

  POLYSTL_LOG("<%s, %d> Host fallback", global, __func__);
  for (int64_t i = 0; i < global; ++i) {
    f(i);
  }
}

} // namespace

template <class ExecutionPolicy, class ForwardIt, class UnaryFunction>
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, void> //
for_each(ExecutionPolicy &&, ForwardIt first, ForwardIt last, UnaryFunction f) {                    // NOLINT(*-dcl58-cpp)
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    std::for_each(first, last, f);
    return;
  }
  parallel_for(std::distance(first, last), [f, first](auto idx) { f(*(first + idx)); });
}

template <class ExecutionPolicy, class ForwardIt, class T>                                          //
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, void> //
fill(ExecutionPolicy &&e, ForwardIt first, ForwardIt last, const T &value) {
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    std::fill(first, last, value);
    return;
  }
  // FIXME we need `value = value` to drop the reference from T&, but this should work without this
  parallel_for(std::distance(first, last), [value = value, first](auto idx) { (*(first + idx)) = value; });
  // FIXME the following version of the lambda lowers to Ptr[T] := Ptr[T] which is wrong
  //  std::for_each(first, last, [&](auto &x) { x = value; });
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, ForwardIt2> //
copy(ExecutionPolicy &&, ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first) {
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    return std::copy(first, last, d_first);
  }
  parallel_for(std::distance(first, last), [d_first, first](auto idx) { (*(d_first + idx)) = (*(first + idx)); });
  return d_first;
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class UnaryOperation>                //
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, ForwardIt2> //
transform(ExecutionPolicy &&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 d_first, UnaryOperation unary_op) {
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    return std::transform(first1, last1, d_first, unary_op);
  }
  parallel_for(std::distance(first1, last1), [d_first, first1, unary_op](auto idx) { (*(d_first + idx)) = unary_op(*(first1 + idx)); });
  return d_first;
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class ForwardIt3, class BinaryOperation> //
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, ForwardIt3>     //
transform(ExecutionPolicy &&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt3 d_first, BinaryOperation binary_op) {
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    return std::transform(first1, last1, first2, d_first, binary_op);
  }
  // TODO
  return d_first;
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T>                    //
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, T> //
transform_reduce(ExecutionPolicy &&e, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init) {
  return transform_reduce(std::forward<ExecutionPolicy &&>(e), first1, last1, first2, init, std::plus<>(), std::multiplies<>());
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T, class BinaryReductionOp,
          class BinaryTransformOp>                                                               //
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, T> //
transform_reduce(ExecutionPolicy &&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init, BinaryReductionOp reduce,
                 BinaryTransformOp transform) {
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    return std::transform_reduce(first1, last1, first2, init, reduce, transform);
  }
  // TODO
  return T{};
}

template <class ExecutionPolicy, class ForwardIt, class T, class BinaryReductionOp,
          class UnaryTransformOp>                                                                //
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, T> //
transform_reduce(ExecutionPolicy &&, ForwardIt first, ForwardIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform) {
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    return std::transform_reduce(first, last, init, reduce, transform);
  }
  // TODO
  return T{};
}

} // namespace std
