#pragma once

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
  #define NOMINMAX
  #include <Windows.h>
  #include <system_error>

  #define polyregion_dl_open(path) LoadLibraryA(path)
  #define polyregion_dl_error() std::system_category().message(::GetLastError()).c_str()
  #define polyregion_dl_close(lib) FreeLibrary(lib)
  #define polyregion_dl_find(lib, symbol) GetProcAddress(lib, symbol)
  #define polyregion_dl_handle HMODULE
#else
  #include <dlfcn.h>

  #ifdef _AIX
    #define polyregion_dl_open(path) dlopen(path, RTLD_MEMBER | RTLD_LAZY | RTLD_GLOBAL)
  #else
    #define polyregion_dl_open(path) dlopen(path, RTLD_LAZY)
  #endif

  #define polyregion_dl_error() dlerror()
  #define polyregion_dl_close(lib) dlclose(lib)
  #define polyregion_dl_find(lib, symbol) dlsym(lib, symbol)
  #define polyregion_dl_handle void *

#endif
