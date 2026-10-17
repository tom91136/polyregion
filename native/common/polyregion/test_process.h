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

inline void defaultCacheDirsToCwd() {
  llvm::SmallString<256> cwd;
  if (llvm::sys::fs::current_path(cwd)) return;
  const auto under = [&](const char *name) {
    llvm::SmallString<256> dir(cwd);
    llvm::sys::path::append(dir, name);
    return dir;
  };
  env::put(env::PolyregionCacheDir, under("module-cache").c_str(), false);
  env::put("POCL_CACHE_DIR", under("pocl-cache").c_str(), false);
}

} // namespace polyregion
