#include "memoryfs.h"

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace polyregion::memoryfs {

std::optional<std::string> open(const std::string &name) {
#ifdef _WIN32
  std::wstring wname(name.begin(), name.end());
  HANDLE h =
      CreateFileW(wname.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) return {};
  CloseHandle(h);
  return name;
#else
  auto ns = "/" + name;
  int shmFd = shm_open(ns.data(), O_RDWR | O_CREAT, S_IRWXU);
  if (shmFd < 0) {
    auto tmp = name + "-XXXXXX";
    int tmpFd = mkstemp(tmp.data());
    if (tmpFd < 0) return {};
    ::close(tmpFd);
    return tmp;
  }
  ::close(shmFd);
  return "/dev/shm" + ns;
#endif
}

bool close(const std::string &name) {
#ifdef _WIN32
  std::wstring wname(name.begin(), name.end());
  return DeleteFileW(wname.c_str());
#else
  auto ns = "/" + name;
  if (shm_unlink(ns.data()) != 0) {
    return unlink(ns.data()) == 0;
  }
  return true;
#endif
}

} // namespace polyregion::memoryfs