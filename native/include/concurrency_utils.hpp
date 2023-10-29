#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <cmath>

namespace polyregion::concurrency_utils {

template <typename F> static void waitAll(F f, size_t timeoutMillis = 10000) {
  std::atomic_size_t pending(1);
  std::condition_variable cv;
  std::mutex mutex;

  const auto countdown = [&]() {
    std::lock_guard lock(mutex);
    pending--;
    cv.notify_all();
  };
  auto now = std::chrono::system_clock::now();
  f(countdown);
  std::unique_lock lock(mutex);
  cv.wait_until(lock, now + std::chrono::milliseconds(timeoutMillis), [&]() { return pending == 0; });
}

template <typename... Fs> static void waitAll(size_t timeoutMillis = 10000, Fs &&...fs) {
  ([&] { waitAll(fs, timeoutMillis); }(), ...);
}

template <typename F> static void waitAllN(F f, size_t timeoutMillis = 10000) {
  std::atomic_size_t pending{0};
  std::condition_variable cv;
  auto now = std::chrono::system_clock::now();
  std::mutex mutex;

  f([&]() {
    pending++;
    return [&]() {
      std::lock_guard lock(mutex);
      pending--;
      cv.notify_all();
    };
  });
  std::unique_lock<std::mutex> lock(mutex);
  cv.wait_until(lock, now + std::chrono::milliseconds(timeoutMillis), [&]() { return pending == 0; });
}

template <typename T = int64_t>
std::vector<std::pair<T, T>> splitStaticExclusive(T start, T end, T N) {
  assert(N >= 0);
  auto range = std::abs(end - start);
  if (range == 0) return {};
  else if (N == 1)
    return {{start, end}};
  else if (range < N) {
    std::vector<std::pair<T, T>> xs(range);
    for (T i = 0; i < range; ++i)
      xs[i] = {start + i, 1};
    return xs;
  } else {
    std::vector<std::pair<T, T>> xs(N);
    auto k = range / N;
    auto m = range % N;
    for (int64_t i = 0; i < N; ++i) {
      auto a = i * k + std::min(i, m);
      auto b = (i + 1) * k + std::min((i + 1), m);
      xs[i] = {start + a, start + b};
    }
    return xs;
  }
}

template <typename T>
constexpr std::pair<std::vector<T>, std::vector<T>> sequencePair(std::vector<std::pair<T, T>> xs) {
  std::pair<std::vector<T>, std::vector<T>> out(std::vector<T>(xs.size()), std::vector<T>(xs.size()));
  for (size_t i = 0; i < xs.size(); ++i) {
    std::get<0>(out)[i] = std::get<0>(xs[i]);
    std::get<1>(out)[i] = std::get<1>(xs[i]);
  }
  return out;
}

} // namespace polyregion::concurrency_utils
