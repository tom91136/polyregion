#pragma once

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

namespace polyregion::env {

inline void put(const char *name, const char *value, bool replace) {
#ifdef _WIN32
  if (!replace && getenv(name)) return;
  _putenv_s(name, value);
#else
  setenv(name, value, replace ? 1 : 0);
#endif
}

inline void unset(const char *name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

class ScopedEnv {
  std::string name;
  std::optional<std::string> previous;

public:
  ScopedEnv(std::string name, const char *value) : name(std::move(name)) {
    if (const char *old = getenv(this->name.c_str())) previous = old;
    put(this->name.c_str(), value, true);
  }
  ScopedEnv(const ScopedEnv &) = delete;
  ScopedEnv &operator=(const ScopedEnv &) = delete;
  ~ScopedEnv() {
    if (previous) put(name.c_str(), previous->c_str(), true);
    else unset(name.c_str());
  }
};

} // namespace polyregion::env
