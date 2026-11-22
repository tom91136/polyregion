#pragma once

#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
  // Must precede the first Darwin dlfcn.h inclusion or RTLD_DEFAULT/RTLD_NEXT stay hidden.
  #define _DARWIN_C_SOURCE
#endif

#include <cstdlib>
#include <string>

#include "compat.h"

#ifdef _WIN32
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

inline std::string polyregion_dl_module_path([[maybe_unused]] const void *address) {
#ifdef _WIN32
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(address), &module))
    return {};
  std::wstring wide(32767, L'\0');
  const auto length = GetModuleFileNameW(module, wide.data(), static_cast<DWORD>(wide.size()));
  if (!length || length >= wide.size()) return {};
  const auto narrowSize = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(length), nullptr, 0, nullptr, nullptr);
  std::string narrow(static_cast<size_t>(narrowSize), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(length), narrow.data(), narrowSize, nullptr, nullptr);
  return narrow;
#elif defined(__linux__) || defined(__APPLE__)
  Dl_info info;
  if (!dladdr(address, &info) || !info.dli_fname) return {};
  auto *const resolved = realpath(info.dli_fname, nullptr);
  if (!resolved) return {};
  std::string path(resolved);
  free(resolved);
  return path;
#else
  return {};
#endif
}
