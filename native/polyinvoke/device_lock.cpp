#include "polyinvoke/device_lock.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "magic_enum/magic_enum.hpp"

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

std::string sanitise(std::string_view in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
      out.push_back(c);
    } else if (!out.empty() && out.back() != '_') {
      out.push_back('_');
    }
  }
  while (!out.empty() && out.back() == '_')
    out.pop_back();
  if (out.empty()) out = "device";
  return out;
}

std::string lockPath(Backend backend, std::string_view deviceName) {
  std::error_code ec;
  auto dir = std::filesystem::temp_directory_path(ec);
  if (ec) dir = std::filesystem::current_path();
  auto key = std::string(magic_enum::enum_name(backend));
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
  if (key.size() > 6 && key.compare(key.size() - 6, 6, "object") == 0) key.resize(key.size() - 6);
  return (dir / ("polyinvoke-" + key + "-" + sanitise(deviceName) + ".lock")).string();
}

} // namespace

#ifdef _WIN32

struct DeviceLock::Impl {
  HANDLE hFile = INVALID_HANDLE_VALUE;
  std::string path;
};

DeviceLock::DeviceLock(Backend backend, std::string_view deviceName) : impl_(std::make_unique<Impl>()) {
  impl_->path = lockPath(backend, deviceName);
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

DeviceLock::DeviceLock(Backend backend, std::string_view deviceName) : impl_(std::make_unique<Impl>()) {
  impl_->path = lockPath(backend, deviceName);
  impl_->fd = ::open(impl_->path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (impl_->fd < 0) {
    throw std::runtime_error("DeviceLock: cannot open lock file " + impl_->path + " (errno=" + std::to_string(errno) + ")");
  }
  if (::flock(impl_->fd, LOCK_EX) != 0) {
    int err = errno;
    ::close(impl_->fd);
    impl_->fd = -1;
    throw std::runtime_error("DeviceLock: flock failed on " + impl_->path + " (errno=" + std::to_string(err) + ")");
  }
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
