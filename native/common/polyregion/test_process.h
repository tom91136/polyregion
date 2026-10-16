#pragma once

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#ifndef _WIN32
  #include <sys/resource.h>
#endif

#include "polyregion/env.h"
#include "polyregion/env_keys.h"

namespace polyregion {

inline void disableCoreDumps() {
#ifndef _WIN32
  const rlimit noCore{0, 0};
  setrlimit(RLIMIT_CORE, &noCore);
#endif
}

inline void defaultCacheDirToCwd() {
  llvm::SmallString<256> dir;
  if (llvm::sys::fs::current_path(dir)) return;
  llvm::sys::path::append(dir, "module-cache");
  env::put(env::PolyregionCacheDir, dir.c_str(), false);
}

} // namespace polyregion
