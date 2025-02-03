#pragma once

#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <vector>

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

template <typename T = int64_t> //
std::pair<std::vector<T>, std::vector<T>> splitStaticExclusive(T start, T end, T N) {
  static_assert(std::is_signed_v<T>, "Numeric type must be signed");
  assert(N >= 0);
  auto range = std::abs(end - start);
  if (range == 0) return {{}, {}};
  else if (N == 1) return {{start}, {end}};
  else if (range < N) {
    std::vector<T> xs(range);
    std::vector<T> ys(range);
    for (T i = 0; i < range; ++i) {
      xs[i] = start + i;
      ys[i] = start + i + 1;
    }
    return {xs, ys};
  } else {
    std::vector<T> xs(N);
    std::vector<T> ys(N);
    auto k = range / N;
    auto m = range % N;
    for (T i = 0; i < N; ++i) {
      auto a = i * k + std::min(i, m);
      auto b = (i + 1) * k + std::min((i + 1), m);
      xs[i] = start + a;
      ys[i] = start + b;
    }
    return {xs, ys};
  }
}

template <typename T> constexpr T tripCountExclusive(T lowerBound, T upperBound, T step) {
  if (lowerBound >= upperBound) return 0;
  return (upperBound - lowerBound + (step - 1)) / step;
}

template <typename T> constexpr T zeroOffset(T induction, T lowerBound, T step) { return lowerBound + (induction * step); }

} // namespace polyregion::concurrency_utils
