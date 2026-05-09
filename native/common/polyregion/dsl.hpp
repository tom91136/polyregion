#pragma once

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/aliases.h"

namespace polyregion {

template <typename E> [[nodiscard]] constexpr StringView enum_name(E e) noexcept { //
  return magic_enum::enum_name(e);
}
template <typename E, typename V> [[nodiscard]] constexpr Opt<E> enum_cast(V v) noexcept { //
  return magic_enum::enum_cast<E>(v);
}

} // namespace polyregion
