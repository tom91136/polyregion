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

POLYREGION_EXAMPLE_ANNOTATE("polyregion_interface:example:example.array") //
inline void array(std::int32_t (&values)[4]) {
  POLYREGION_EXAMPLE_IMPLEMENT(array, values);
}

} // namespace example

#undef POLYREGION_EXAMPLE_ANNOTATE

#pragma pop_macro("POLYREGION_EXAMPLE_IMPLEMENT")
