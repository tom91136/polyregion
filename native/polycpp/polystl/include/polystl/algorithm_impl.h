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

template <class ExecutionPolicy, class ForwardIt, class UnaryFunction>
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, void> //
for_each(ExecutionPolicy &&, ForwardIt first, ForwardIt last, UnaryFunction f) {                    // NOLINT(*-dcl58-cpp)

  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    std::for_each(first, last, f);
    return;
  }

  auto global = std::distance(first, last);
  auto N = std::thread::hardware_concurrency();
  std::fprintf(stderr, "[POLYSTL:%s] Dispatch global range <%d>\n", __func__, global);

  if (auto kind = polystl::platformKind(); kind) {
    switch (*kind) {
      case polyregion::runtime::PlatformKind ::HostThreaded: {
        auto [b, e] = polyregion::concurrency_utils::splitStaticExclusive2<int64_t>(0, global, 1);
        const int64_t *begin = b.data();
        const int64_t *end = e.data();
        const auto kernel = [&f, &first, begin, end](const int64_t tid) {
          for (int64_t i = begin[tid]; i < end[tid]; ++i) {
            f(*(first + i));
          }
        };
        auto &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::HostThreaded>(kernel);
        std::byte argData[sizeof(decltype(kernel))];
        std::memcpy(argData, &kernel, sizeof(decltype(kernel)));
        for (auto &object : bundle.objects) {
          if (polystl::dispatchHostThreaded(b.size(), &argData, object)) return;
        }
        break;
      }
      case polyregion::runtime::PlatformKind ::Managed: {
        const auto kernel = [&f, &first]() { f(*(first + __polyregion_builtin_gpu_global_idx(0))); };
        auto &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::Managed>(kernel);
        for (auto &object : bundle.objects) {
          if (polystl::dispatchManaged(global, 0, 0, sizeof(decltype(kernel)), &kernel, object)) return;
        }
        break;
      }
    }
  }

  std::fprintf(stderr, "[POLYSTL:%s] Host fallback\n", __func__);
  for (size_t i = 0; i < global; ++i) {
    f(*(first + i));
  }
}

} // namespace std
