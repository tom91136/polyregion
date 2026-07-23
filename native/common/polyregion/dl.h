#pragma once

#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
  // Must precede the first Darwin dlfcn.h inclusion or RTLD_DEFAULT/RTLD_NEXT stay hidden.
  #define _DARWIN_C_SOURCE
#endif

#include "compat.h"

#ifdef _WIN32
  #include <string>
  #include <system_error>

  #include <Windows.h>

inline const char *polyregion_dl_error_() {
  const DWORD code = ::GetLastError();
  if (!code) return nullptr;
  thread_local std::string msg;
  msg = std::system_category().message(code);
  return msg.c_str();
}

  #define polyregion_dl_open(path) LoadLibraryA(path)
  #define polyregion_dl_error() polyregion_dl_error_()
  #define polyregion_dl_close(lib) (FreeLibrary(lib) ? 0 : 1)
  #define polyregion_dl_find(lib, symbol) GetProcAddress(lib, symbol)
  #define polyregion_dl_handle HMODULE
#else
  #include <dlfcn.h>

  #ifdef _AIX
    #define polyregion_dl_open(path) dlopen(path, RTLD_MEMBER | RTLD_LAZY | RTLD_GLOBAL)
  #else
    #define polyregion_dl_open(path) dlopen(path, RTLD_LAZY | RTLD_LOCAL)
  #endif

  #define polyregion_dl_error() dlerror()
  #define polyregion_dl_close(lib) dlclose(lib)
  #define polyregion_dl_find(lib, symbol) dlsym(lib, symbol)
  #define polyregion_dl_handle void *

#endif
