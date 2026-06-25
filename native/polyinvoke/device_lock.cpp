// _POSIX_C_SOURCE hides BSD flock/LOCK_* on macOS; must define _DARWIN_C_SOURCE
// before any system header is pulled in transitively.
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
  #define _DARWIN_C_SOURCE
#endif

#include "polyinvoke/device_lock.h"

#include <stdexcept>
#include <string>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <unistd.h>

  #include <sys/file.h>
#endif

namespace polyregion::invoke {

namespace {

std::string lockPath(const PhysicalDevice &device) {
  llvm::SmallString<256> dir;
  llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, dir);
  if (dir.empty()) {
    if (auto ec = llvm::sys::fs::current_path(dir)) dir = ".";
  }
  llvm::sys::path::append(dir, "polyinvoke-" + device.str() + ".lock");
  return dir.str().str();
}

} // namespace

#ifdef _WIN32

struct DeviceLock::Impl {
  HANDLE hFile = INVALID_HANDLE_VALUE;
  std::string path;
};

DeviceLock::DeviceLock(const PhysicalDevice &device) : impl_(std::make_unique<Impl>()) {
  if (!device.needsLock()) return;
  impl_->path = lockPath(device);
  impl_->hFile = CreateFileA(impl_->path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
  if (impl_->hFile == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("DeviceLock: cannot open lock file " + impl_->path + " (GetLastError=" + std::to_string(GetLastError()) + ")");
  }
  OVERLAPPED ov{};
  if (!LockFileEx(impl_->hFile, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &ov)) {
    DWORD err = GetLastError();
    CloseHandle(impl_->hFile);
    impl_->hFile = INVALID_HANDLE_VALUE;
    throw std::runtime_error("DeviceLock: LockFileEx failed on " + impl_->path + " (GetLastError=" + std::to_string(err) + ")");
  }
}

DeviceLock::~DeviceLock() {
  if (impl_ && impl_->hFile != INVALID_HANDLE_VALUE) {
    OVERLAPPED ov{};
    UnlockFileEx(impl_->hFile, 0, MAXDWORD, MAXDWORD, &ov);
    CloseHandle(impl_->hFile);
  }
}

#else

struct DeviceLock::Impl {
  int fd = -1;
  std::string path;
};

DeviceLock::DeviceLock(const PhysicalDevice &device) : impl_(std::make_unique<Impl>()) {
  if (!device.needsLock()) return;
  impl_->path = lockPath(device);
  int fd = ::open(impl_->path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) {
    throw std::runtime_error("DeviceLock: cannot open lock file " + impl_->path + " (errno=" + std::to_string(errno) + ")");
  }
  if (::flock(fd, LOCK_EX) != 0) {
    int err = errno;
    ::close(fd);
    throw std::runtime_error("DeviceLock: flock failed on " + impl_->path + " (errno=" + std::to_string(err) + ")");
  }
  if (int hi = ::fcntl(fd, F_DUPFD_CLOEXEC, 900); hi >= 0) {
    ::close(fd);
    impl_->fd = hi;
  } else impl_->fd = fd;
}

DeviceLock::~DeviceLock() {
  if (impl_ && impl_->fd >= 0) {
    ::flock(impl_->fd, LOCK_UN);
    ::close(impl_->fd);
  }
}

#endif

DeviceLock::DeviceLock(DeviceLock &&) noexcept = default;
DeviceLock &DeviceLock::operator=(DeviceLock &&) noexcept = default;

} // namespace polyregion::invoke
