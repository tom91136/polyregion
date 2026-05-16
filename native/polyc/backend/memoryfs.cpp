#include "memoryfs.h"

#include <string>

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

// XXX prefix with pid: /dev/shm is process-global and out.<n> collides across polyc processes.
static std::string pidScopedName(const std::string &name) {
#ifdef _WIN32
  return "polyc-" + std::to_string(static_cast<unsigned long long>(GetCurrentProcessId())) + "-" + name;
#else
  return "polyc-" + std::to_string(static_cast<unsigned long long>(getpid())) + "-" + name;
#endif
}

std::optional<std::string> open(const std::string &name) {
  const auto scoped = pidScopedName(name);
#ifdef _WIN32
  std::wstring wname(scoped.begin(), scoped.end());
  HANDLE h = CreateFileW(wname.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) return {};
  CloseHandle(h);
  return scoped;
#else
  auto ns = "/" + scoped;
  int shmFd = shm_open(ns.data(), O_RDWR | O_CREAT | O_EXCL, S_IRWXU);
  if (shmFd < 0) {
    auto tmp = scoped + "-XXXXXX";
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
  if (name.rfind("/dev/shm/", 0) == 0) {
    auto leaf = name.substr(std::string("/dev/shm").size());
    if (shm_unlink(leaf.data()) == 0) return true;
  }
  return unlink(name.data()) == 0;
#endif
}

} // namespace polyregion::memoryfs