#pragma once

#include <cstdlib>
#include <optional>
#include <string>

class ScopedEnv {
  const char *name;
  std::optional<std::string> previous;

  static void set(const char *name, const std::optional<std::string> &value) {
#ifdef _WIN32
    _putenv_s(name, value ? value->c_str() : "");
#else
    if (value) setenv(name, value->c_str(), 1);
    else unsetenv(name);
#endif
  }

public:
  ScopedEnv(const char *name, std::optional<std::string> value) : name(name) {
    if (const auto *v = std::getenv(name)) previous = v;
    set(name, value);
  }
  ~ScopedEnv() { set(name, previous); }
};
