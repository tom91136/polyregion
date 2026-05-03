#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace polyregion {

[[noreturn]] inline void raise(const std::string &message, const char *file = __builtin_FILE(), int line = __builtin_LINE()) {
  std::fprintf(stderr, "[%s:%d] %s\n", file, line, message.c_str());
  std::fflush(stderr);
  std::abort();
}

} // namespace polyregion

#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
  #define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#define POLYREGION_FATAL(prefix, fmt, ...)                                                                                                 \
  do {                                                                                                                                     \
    std::fprintf(stderr, "[%s] %s:%d FATAL: " fmt " (in `%s`)\n", (prefix), __FILE__, __LINE__, __VA_ARGS__, __PRETTY_FUNCTION__);         \
    std::fflush(stderr);                                                                                                                   \
    std::abort();                                                                                                                          \
  } while (0)
