#include "memoryfs.h"

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <unistd.h>

  #include <sys/mman.h>
  #include <sys/stat.h>
#endif

namespace polyregion::memoryfs {

std::optional<std::string> open(const std::string &name) {
#ifdef _WIN32
  std::wstring wname(name.begin(), name.end());
  HANDLE h = CreateFileW(wname.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) return {};
  CloseHandle(h);
  return name;
#else
  auto tryMkstemp = [&](const std::string &dir) -> std::optional<std::string> {
    auto tmp = dir + "/" + name + "-XXXXXX";
    if (int fd = mkstemp(tmp.data()); fd < 0) return {};
    else {
      ::close(fd);
      return tmp;
    }
  };
  #ifdef __linux__
  if (auto p = tryMkstemp("/dev/shm")) return p;
  #endif
  return tryMkstemp("/tmp");
#endif
}

bool close(const std::string &name) {
#ifdef _WIN32
  std::wstring wname(name.begin(), name.end());
  return DeleteFileW(wname.c_str());
#else
  return unlink(name.data()) == 0;
#endif
}

} // namespace polyregion::memoryfs