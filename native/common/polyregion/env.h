#pragma once

#include <cstdlib>

namespace polyregion::env {

void put(const char *name, const char *value, bool replace) {
#ifdef _WIN32
  if (!replace && getenv(name)) return;
  _putenv_s(name, value);
#else
  setenv(name, value, replace ? 1 : 0);
#endif
}

} // namespace polyregion::env