#pragma once

#include <cstdint>
#include <type_traits>

namespace example {

template <class T>
[[clang::annotate("polyregion_import:example:example.count")]] inline std::int32_t count(const T *in, std::int32_t n) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:example:example.transform")]] inline void transform(const T *in, T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, T>, T>, "callable signature mismatch");
  __builtin_trap();
}

} // namespace example
