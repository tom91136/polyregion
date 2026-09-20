#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__clang__)
#define POLYREGION_EXAMPLE_ANNOTATE(value) [[clang::annotate(value)]]
#else
#define POLYREGION_EXAMPLE_ANNOTATE(value)
#endif

#pragma push_macro("POLYREGION_EXAMPLE_IMPLEMENT")
#ifndef POLYREGION_EXAMPLE_IMPLEMENT
#define POLYREGION_EXAMPLE_IMPLEMENT(function, ...) __builtin_trap()
#endif

namespace example {

template <class T> //
POLYREGION_EXAMPLE_ANNOTATE("polyregion_interface:example:example.count") //
inline std::int32_t count(const T *in, std::int32_t n) {
  POLYREGION_EXAMPLE_IMPLEMENT(count, in, n);
}

template <class T, class U, class Op> //
POLYREGION_EXAMPLE_ANNOTATE("polyregion_interface:example:example.transform") //
inline void transform(const T *in, U *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, U>, "callable signature mismatch");
  POLYREGION_EXAMPLE_IMPLEMENT(transform, in, out, n, op);
}

} // namespace example

#undef POLYREGION_EXAMPLE_ANNOTATE

#pragma pop_macro("POLYREGION_EXAMPLE_IMPLEMENT")
