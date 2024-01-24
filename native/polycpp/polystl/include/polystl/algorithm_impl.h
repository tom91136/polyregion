#pragma once

#include <type_traits>
#include <execution>

// NOLINTBEGIN(*-dcl58-cpp)
namespace std {

template <class ExecutionPolicy, class ForwardIt, class UnaryFunction2>
typename std::enable_if_t<std::is_execution_policy_v<std::decay_t<ExecutionPolicy>>, void> //
for_each(ExecutionPolicy &&, ForwardIt first, ForwardIt last, UnaryFunction2 f) {
  auto _N = std::distance(first, last);



  __polyregion_offload_dispatch__(N, 0, 0, [](a))


}


} // namespace std
// NOLINTEND(*-dcl58-cpp)

