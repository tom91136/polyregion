#pragma once

#include <string>

#include "llvm/Support/raw_ostream.h"

#include "polyregion/error.h"

namespace polyregion::polyfc {

using polyregion::raise;

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

template <typename Diag, std::size_t N, typename... Args>
void emit(Diag &diag, typename Diag::Level level, const char (&fmt)[N], Args &&...args) {
  auto b = diag.Report(diag.getCustomDiagID(level, fmt));
  (b << ... << std::forward<Args>(args));
}

} // namespace polyregion::polyfc
