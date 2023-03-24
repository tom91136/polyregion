#include "memoryfs.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace polyregion::memoryfs {

std::optional<std::string> open(const std::string &name) {
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
}

bool close(const std::string &name) {
  auto ns = "/" + name;
  if (shm_unlink(ns.data()) != 0) {
    return unlink(ns.data()) == 0;
  }
  return true;
}

} // namespace polyregion::memoryfs