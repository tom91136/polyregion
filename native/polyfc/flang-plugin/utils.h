#pragma once

#include <cstdio>
#include <string>

#include "llvm/Support/raw_ostream.h"

namespace polyregion::polyfc {

[[noreturn]] inline void raise(const std::string &message, const char *file = __builtin_FILE(), int line = __builtin_LINE()) {
  std::fprintf(stderr, "[%s:%d] %s\n", file, line, message.c_str());
  std::fflush(stderr);
  std::abort();
}

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
