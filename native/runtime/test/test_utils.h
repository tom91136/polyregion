#pragma once

#include "runtime.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace polyregion::test_utils {

std::optional<std::string>
findTestImage(const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>> &images,
              const polyregion::runtime::Backend &backend, const std::vector<std::string> &features);

template <typename... Fs> static void waitAll(Fs &&...fs) {
  std::atomic_size_t pending{sizeof...(Fs)};
  std::condition_variable cv;
  std::mutex mutex;

  const auto countdown = [&]() {
    pending--;
    cv.notify_all();
  };

  ([&] { fs(countdown); }(), ...);

  std::unique_lock<std::mutex> lock(mutex);
  cv.wait(lock, [&]() { return pending == 0; });
}

template <typename F> static void waitAllN(size_t N, F f) {
  std::atomic_size_t pending{N};
  std::condition_variable cv;
  std::mutex mutex;
  auto now = std::chrono::system_clock::now();
  f([&]() {
    pending--;
    cv.notify_all();
  });
  std::unique_lock<std::mutex> lock(mutex);
  if(!cv.wait_until(lock, now + std::chrono::milliseconds(10000),[&]() { return pending == 0; })){

  }
}

} // namespace polyregion::test_utils
