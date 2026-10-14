#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__clang__)
  #define POLYREGION_POLYREGION_CORE_ANNOTATE(value) [[clang::annotate(value)]]
#else
  #define POLYREGION_POLYREGION_CORE_ANNOTATE(value)
#endif

#pragma push_macro("POLYREGION_POLYREGION_CORE_IMPLEMENT")
#ifdef POLYREGION_POLYREGION_CORE_IMPLEMENT
  #define POLYREGION_POLYREGION_CORE_FALLBACK POLYREGION_POLYREGION_CORE_ANNOTATE("polyregion_interface_fallback")
#else
  #define POLYREGION_POLYREGION_CORE_FALLBACK
  #if defined(_MSC_VER) && !defined(__clang__)
    #define POLYREGION_POLYREGION_CORE_IMPLEMENT(function, ...) (__debugbreak(), __assume(0))
  #else
    #define POLYREGION_POLYREGION_CORE_IMPLEMENT(function, ...) __builtin_trap()
  #endif
#endif

namespace polyregion::core {

template <class Body>                                                                                                                    //
POLYREGION_POLYREGION_CORE_ANNOTATE("polyregion_interface:polyregion.core:polyregion.core.for_each") POLYREGION_POLYREGION_CORE_FALLBACK //
    inline void for_each(std::int64_t n, Body body) {
  static_assert(std::is_same_v<std::invoke_result_t<Body &, const std::int64_t &>, void>, "callable signature mismatch");
  POLYREGION_POLYREGION_CORE_IMPLEMENT(for_each, n, body);
}

template <class T, class Map, class Combine> //
POLYREGION_POLYREGION_CORE_ANNOTATE("polyregion_interface:polyregion.core:polyregion.core.transform_reduce")
    POLYREGION_POLYREGION_CORE_FALLBACK //
    inline T transform_reduce(std::int64_t n, T init, Map map, Combine combine) {
  static_assert(std::is_same_v<std::invoke_result_t<Map &, const std::int64_t &>, T>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Combine &, const T &, const T &>, T>, "callable signature mismatch");
  POLYREGION_POLYREGION_CORE_IMPLEMENT(transform_reduce, n, init, map, combine);
}

} // namespace polyregion::core

#undef POLYREGION_POLYREGION_CORE_FALLBACK
#undef POLYREGION_POLYREGION_CORE_ANNOTATE

#pragma pop_macro("POLYREGION_POLYREGION_CORE_IMPLEMENT")
