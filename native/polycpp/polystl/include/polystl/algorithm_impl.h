#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#include <type_traits>
#include <vector>

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

template <typename T = int64_t> //
std::pair<std::vector<T>, std::vector<T>> splitStaticExclusive(T start, T end, T N) {
  assert(N >= 0);
  auto range = std::abs(end - start);
  if (range == 0) return {{}, {}};
  else if (N == 1) return {{start}, {end}};
  else if (range < N) {
    std::vector<T> xs(range);
    std::vector<T> ys(range);
    for (T i = 0; i < range; ++i) {
      xs[i] = start + i;
      ys[i] = start + i + 1;
    }
    return {xs, ys};
  } else {
    std::vector<T> xs(N);
    std::vector<T> ys(N);
    auto k = range / N;
    auto m = range % N;
    for (int64_t i = 0; i < N; ++i) {
      auto a = i * k + std::min(i, m);
      auto b = (i + 1) * k + std::min((i + 1), m);
      xs[i] = start + a;
      ys[i] = start + b;
    }
    return {xs, ys};
  }
}

static constexpr const char *HostFallbackEnv = "POLYSTL_HOST_FALLBACK";

inline bool host_fallback() {
  if (auto env = std::getenv(HostFallbackEnv); env) {
    errno = 0; // strtol to avoid exceptions
    size_t value = std::strtol(env, nullptr, 10);
    if (errno == 0 && value == 0) {
      POLYSTL_LOG("<%s> No compatible backend and host fallback disabled, returning...", __func__);
      return false;
    }
  }
  return true; // default is to use host fallback
}

template <class UnaryFunction> void parallel_for(int64_t global, UnaryFunction f) {
  auto N = std::thread::hardware_concurrency();
  POLYSTL_LOG("<%s, %ld> Dispatch", __func__, global);

  if (PlatformKind kind; __polyregion_platform_kind(kind)) {
    switch (kind) {
      case polyregion::runtime::PlatformKind ::HostThreaded: {
        auto [b, e] = splitStaticExclusive<int64_t>(0, global, 1);
        const int64_t *begin = b.data();
        const int64_t *end = e.data();
        const auto kernel = [&f, begin, end](const int64_t tid) {
          for (int64_t i = begin[tid]; i < end[tid]; ++i) {
            f(i);
          }
        };
        fprintf(stderr, "=== \n" );
        const RuntimeKernelBundle &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::HostThreaded>(kernel);


        fprintf(stderr, "=== %d\n", bundle.objects[0].imageLength);
        std::byte argData[sizeof(decltype(kernel))];
        std::memcpy(argData, &kernel, sizeof(decltype(kernel)));
        for (size_t i = 0; i < bundle.objectCount; ++i) {
          if (!__polyregion_load_kernel_object(bundle.moduleName, bundle.objects[i])) continue;
          __polyregion_dispatch_hostthreaded(b.size(), &argData, bundle.moduleName);
          return;
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
        fprintf(stderr, "=== \n" );
        const RuntimeKernelBundle &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::Managed>(kernel);


        // T* p = reflect_get(kernel, "a");
        // auto size = query_ptr(p);
        // auto devicePtr = mallocDevice(size);
        //
        // enqueueAsync(p -> devicePtr, size);
        // reflect_set(kernel, "a", devicePtr);
        // <<<>>>
        // enqueueAsync(devicePtr -> p, size);
        // TLI
        //

        // std::vector< struct Reflected { lambdaPtr: void*, lambdaPtrSize: size_t,  } >
        fprintf(stderr, "=== %s\n", bundle.moduleName);

        for (size_t i = 0; i < bundle.objectCount; ++i) {
          if (!__polyregion_load_kernel_object(bundle.moduleName, bundle.objects[i])) continue;



          // float* a;
          // linked<int> x { linked<int>* next; };



          struct Reflected {
             std::tuple<const char*, char *, size_t>* topLevelPointers;

             std::tuple<const char*, uintptr_t, size_t >* arbitraryPointers;
          };
          // __polyregion_runtime_reflect(kernel, [](const char* name, char* data, size_t size, bool indirect) -> bool {
          //
          // });

          //

          __polyregion_dispatch_managed(global / blocks, blocks, 0, sizeof(decltype(kernel)), &kernel, bundle.moduleName);
          return;
        }
        break;
      }
    }
  }

  if (!host_fallback()) return;
  POLYSTL_LOG("<%s, %d> Host fallback", __func__, global);
  for (int64_t i = 0; i < global; ++i) {
    f(i);
  }
}

template <typename T, class UnaryFunction, class BinaryFunction>
T parallel_reduce(int64_t global, T init, UnaryFunction f, BinaryFunction reduce) {
  auto N = std::thread::hardware_concurrency();
  POLYSTL_LOG("<%s, %d> Dispatch", __func__, global);

  if (PlatformKind kind; __polyregion_platform_kind(kind)) {
    switch (kind) {
      case polyregion::runtime::PlatformKind ::HostThreaded: {
        auto [b, e] = splitStaticExclusive<int64_t>(0, global, N);
        const int64_t groups = b.size();
        const int64_t *begin = b.data();
        const int64_t *end = e.data();

        std::vector<T> groupPartial(groups);
        const auto kernel = [&f, &reduce, init, begin, end, out = groupPartial.data()](const int64_t tid) {
          auto acc = init;
          for (int64_t i = begin[tid]; i < end[tid]; ++i) {
            acc = reduce(acc, f(i));
          }
          out[tid] = acc;
        };
        const RuntimeKernelBundle &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::HostThreaded>(kernel);
        std::byte argData[sizeof(decltype(kernel))];
        std::memcpy(argData, &kernel, sizeof(decltype(kernel)));
        for (size_t i = 0; i < bundle.objectCount; ++i) {
          if (!__polyregion_load_kernel_object(bundle.moduleName, bundle.objects[i])) continue;
          __polyregion_dispatch_hostthreaded(groups, &argData, bundle.moduleName);
          T acc = init;
          for (int64_t groupIdx = 0; groupIdx < groups; ++groupIdx) {
            acc = reduce(acc, groupPartial[groupIdx]);
          }
          return acc;
        }
        break;
      }
      case polyregion::runtime::PlatformKind ::Managed: {

        int64_t groups = 256;

        auto groupPartial = __polyregion_selected_device->mallocSharedTyped<T>(groups, Access::RW);

        if (!groupPartial) {
          POLYSTL_LOG("<%s, %d> No USM support", __func__, global);
          std::abort();
        }

        const auto kernel = [groupPartial = *groupPartial, init, f, reduce,
                             global]([[clang::annotate("__polyregion_local")]] T *localPartialSum) {
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
            groupPartial[__polyregion_builtin_gpu_group_idx(0)] = localPartialSum[localIdx];
          }
        };
        const RuntimeKernelBundle &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::Managed>(kernel);
        for (size_t i = 0; i < bundle.objectCount; ++i) {
          if (!__polyregion_load_kernel_object(bundle.moduleName, bundle.objects[i])) continue;
          __polyregion_dispatch_managed(256, groups, groups * sizeof(T), sizeof(decltype(kernel)), &kernel, bundle.moduleName);
          T acc = init;
          for (int64_t groupIdx = 0; groupIdx < groups; ++groupIdx) {
            acc = reduce(acc, (*groupPartial)[groupIdx]);
          }
          return acc;
        }
        break;
      }
    }
  }

  if (!host_fallback()) return init;
  POLYSTL_LOG("<%s, %ld> Host fallback", __func__, global);

  T acc = init;
  for (int64_t globalIdx = 0; globalIdx < global; ++globalIdx) {
    acc = reduce(acc, f(globalIdx));
  }
  return acc;
}

} // namespace details

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
  parallel_for(std::distance(first1, last1),
               [d_first, first1, first2, binary_op](auto idx) { (*(d_first + idx)) = binary_op(*(first1 + idx), *(first2 + idx)); });

  return d_first;
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T>                    //
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, T> //
transform_reduce(ExecutionPolicy &&e, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init) {
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    return std::transform_reduce(first1, last1, first2, init);
  }
  return parallel_reduce(
      std::distance(first1, last1), init, //
      [first1, first2](auto idx) { return (*(first1 + idx)) * (*(first2 + idx)); }, [](auto l, auto r) { return l + r; });
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T, class BinaryReductionOp,
          class BinaryTransformOp>                                                               //
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, T> //
transform_reduce(ExecutionPolicy &&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init, BinaryReductionOp reduce,
                 BinaryTransformOp transform) {
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    return std::transform_reduce(first1, last1, first2, init, reduce, transform);
  }
  return parallel_reduce(
      std::distance(first1, last1), init, //
      [transform, first1, first2](auto idx) { return transform(*(first1 + idx), *(first2 + idx)); }, reduce);
}

template <class ExecutionPolicy, class ForwardIt, class T, class BinaryReductionOp,
          class UnaryTransformOp>                                                                //
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, T> //
transform_reduce(ExecutionPolicy &&, ForwardIt first, ForwardIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform) {
  if constexpr (!std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_unsequenced_policy>) {
    return std::transform_reduce(first, last, init, reduce, transform);
  }
  return parallel_reduce(std::distance(first, last), init, transform, reduce);
}

} // namespace std
