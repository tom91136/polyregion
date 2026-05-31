#pragma once

#include <string>

#include "llvm/Support/raw_ostream.h"

#include "polyfront/diag.hpp"
#include "polyregion/error.h"

namespace polyregion::polyfc {

using polyregion::raise;
using polyregion::polyfront::emit;

template <typename T> auto value_of(T t) { return static_cast<std::underlying_type_t<T>>(t); }

template <typename T> std::string show(T t) {
  std::string result{};
  llvm::raw_string_ostream s(result);
  if constexpr (std::is_pointer_v<T>) {
    if (!t) s << "(nullptr)";
    else t->print(s);
  } else t.print(s);
  return result;
}

} // namespace polyregion::polyfc
