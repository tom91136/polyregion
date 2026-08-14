#pragma once

#include <cstdint>
#include <type_traits>

namespace example {

[[clang::annotate("polyregion_interface:example:example.array")]] inline void array(std::int32_t (&values)[4]) {
  __builtin_trap();
}

} // namespace example
