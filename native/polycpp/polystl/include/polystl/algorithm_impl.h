#pragma once

#include "runtime.h"
#include <thread>
#include <type_traits>

// #include_next <execution>

// Taken from https://en.cppreference.com/w/cpp/header/execution

namespace std::execution { // NOLINT(*-dcl58-cpp)
// sequenced execution policy
class sequenced_policy {};

// parallel execution policy
class parallel_policy {};

// parallel and unsequenced execution policy
class parallel_unsequenced_policy {};

// unsequenced execution policy
class unsequenced_policy {};

// execution policy objects
inline constexpr sequenced_policy seq{};
inline constexpr parallel_policy par{};
inline constexpr parallel_unsequenced_policy par_unseq{};
inline constexpr unsequenced_policy unseq{};
} // namespace std::execution

namespace std {
template <typename> struct is_execution_policy : std::false_type {};                                     // NOLINT(*-dcl58-cpp)
template <> struct is_execution_policy<std::execution::sequenced_policy> : std::true_type {};            // NOLINT(*-dcl58-cpp)
template <> struct is_execution_policy<std::execution::parallel_policy> : std::true_type {};             // NOLINT(*-dcl58-cpp)
template <> struct is_execution_policy<std::execution::parallel_unsequenced_policy> : std::true_type {}; // NOLINT(*-dcl58-cpp)
template <> struct is_execution_policy<std::execution::unsequenced_policy> : std::true_type {};          // NOLINT(*-dcl58-cpp)
template <typename T> constexpr bool is_execution_policy_v = is_execution_policy<T>::value;              // NOLINT(*-dcl58-cpp)
} // namespace std

// NOLINTBEGIN(*-dcl58-cpp)
namespace std {

enum class kind {
  CPU, GPU
};

struct kernel_object{
  size_t imageSize;
  const unsigned char* imageBytes;
  const char * kernelName;
  bool cpu;
  const char** features;

  // void * data;
  // ArgBuffer buffer;
};


template <class ExecutionPolicy, class ForwardIt, class UnaryFunction2>
typename std::enable_if_t<std::is_execution_policy_v<typename std::decay_t<ExecutionPolicy>>, void> //
for_each(ExecutionPolicy &&, ForwardIt first, ForwardIt last, UnaryFunction2 f) {
  auto global_range = std::distance(first, last);

  constexpr static int objects_count = 3;
  static kernel_object objects[objects_count] = {
//    kernel_object(42, {0x00}, "kernel", true, {"sse"})
  };

  for (int i = 0; i < 3; ++i) {
    objects[i].features
    objects[i].features

  }

  for(size_t i = 0; i < global_range;++i){
    f(*(first + i));
  }

  bool cpu = true;

  if (cpu) {

    auto nproc = std::thread::hardware_concurrency();


    polystl::__polyregion_offload_dispatch__(nproc, 0, 0, []() {

    });

    [&f, &first, global_range](const int64_t id, const int64_t *begin, const int64_t *end) {
      for (int64_t i = begin[id]; i < end[id]; ++i) {
        f(*(first + i));
      }
    };
  } else {
    [&f, &first]() { f(*(first + __polyregion__gpu_global_idx(0))); };
  }

//  __polyregion_offload_dispatch__(global_range, 0, 0, []() {
//
//  });
}


} // namespace std
// NOLINTEND(*-dcl58-cpp)

