#pragma once

#include <initializer_list>

#include "polyregion/dl.h"

namespace polyregion::invoke::dl {

inline void *open_first(std::initializer_list<const char *> paths) {
  for (const char *p : paths)
    if (auto *h = polyregion_dl_open(p)) return h;
  return nullptr;
}

inline void *lookup(void *user, const char *name) {
  // NOLINTNEXTLINE(readability-redundant-casting)
  return reinterpret_cast<void *>(polyregion_dl_find(static_cast<polyregion_dl_handle>(user), name));
}

} // namespace polyregion::invoke::dl
