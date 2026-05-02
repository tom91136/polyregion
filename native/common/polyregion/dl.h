#pragma once

#include "compat.h"

#ifdef _WIN32
  #include <Windows.h>
  #include <string>
  #include <system_error>

  inline const char *polyregion_dl_error_() {
    thread_local std::string msg;
    msg = std::system_category().message(::GetLastError());
    return msg.c_str();
  }

  #define polyregion_dl_open(path) LoadLibraryA(path)
  #define polyregion_dl_error() polyregion_dl_error_()
  #define polyregion_dl_close(lib) FreeLibrary(lib)
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
