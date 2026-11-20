#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "polystl/algorithm_impl.h"

namespace polyregion::offload {

namespace detail {

inline std::int64_t extent(std::int64_t a, std::int64_t b) {
  std::int64_t product = 0;
  if (__builtin_mul_overflow(a, b, &product)) throw std::overflow_error("extent product overflows 64 bits");
  return product;
}

} // namespace detail

class Context {
  void *handle_ = nullptr;

public:
  struct Device {
    std::int32_t platform, device;
    std::string platformName, name;
  };

  static std::vector<Device> devices() {
    std::int32_t platforms = 0;
    if (!polyrt_platform_count(&platforms)) throw std::runtime_error(polyrt_error_message());
    std::vector<Device> found;
    for (std::int32_t platform = 0; platform < platforms; ++platform) {
      PolyrtPlatformInfo platformInfo{};
      if (!polyrt_platform_info(platform, &platformInfo)) throw std::runtime_error(polyrt_error_message());
      for (std::int32_t device = 0; device < platformInfo.deviceCount; ++device) {
        PolyrtDeviceInfo deviceInfo{};
        if (!polyrt_device_info(platform, device, &deviceInfo)) throw std::runtime_error(polyrt_error_message());
        found.push_back({platform, device, platformInfo.name, deviceInfo.name});
      }
    }
    return found;
  }

  Context(std::int32_t platform, std::int32_t device) {
    if (!polyrt_context_create(platform, device, &handle_)) throw std::runtime_error(polyrt_error_message());
  }
  explicit Context(const Device &device) : Context(device.platform, device.device) {}
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Context &operator=(Context &&other) noexcept {
    std::swap(handle_, other.handle_);
    return *this;
  }
  ~Context() {
    if (handle_ && !polyrt_context_destroy(handle_)) std::fprintf(stderr, "polyregion::offload::Context: %s\\n", polyrt_error_message());
  }

  void *handle() const { return handle_; }
};

template <class Body> inline void for_each(std::int64_t n, Body body) { ::polyregion::core::for_each(n, body); }

template <class T, class Map, class Combine> inline T transform_reduce(std::int64_t n, T init, Map map, Combine combine) {
  return ::polyregion::core::transform_reduce(n, init, map, combine);
}

template <class Body> inline void for_each(const Context &context, std::int64_t n, Body body) {
  ::polyregion::core::for_each(context.handle(), n, body);
}

template <class T, class Map, class Combine>
inline T transform_reduce(const Context &context, std::int64_t n, T init, Map map, Combine combine) {
  return ::polyregion::core::transform_reduce(context.handle(), n, init, map, combine);
}

template <class Body> inline void for_each(std::int64_t n0, std::int64_t n1, Body body) {
  for_each(detail::extent(n0, n1), [=](std::int64_t i) { return body(i / n1, i % n1); });
}

template <class Body> inline void for_each(const Context &context, std::int64_t n0, std::int64_t n1, Body body) {
  for_each(context, detail::extent(n0, n1), [=](std::int64_t i) { return body(i / n1, i % n1); });
}

template <class Body> inline void for_each(std::int64_t n0, std::int64_t n1, std::int64_t n2, Body body) {
  for_each(detail::extent(detail::extent(n0, n1), n2), [=](std::int64_t i) { return body(i / (n1 * n2), i / n2 % n1, i % n2); });
}

template <class Body> inline void for_each(const Context &context, std::int64_t n0, std::int64_t n1, std::int64_t n2, Body body) {
  for_each(context, detail::extent(detail::extent(n0, n1), n2), [=](std::int64_t i) { return body(i / (n1 * n2), i / n2 % n1, i % n2); });
}

template <class T, class Map, class Combine> inline T transform_reduce(std::int64_t n0, std::int64_t n1, T init, Map map, Combine combine) {
  return transform_reduce(detail::extent(n0, n1), init, [=](std::int64_t i) { return map(i / n1, i % n1); }, combine);
}

template <class T, class Map, class Combine>
inline T transform_reduce(const Context &context, std::int64_t n0, std::int64_t n1, T init, Map map, Combine combine) {
  return transform_reduce(context, detail::extent(n0, n1), init, [=](std::int64_t i) { return map(i / n1, i % n1); }, combine);
}

template <class T, class Map, class Combine>
inline T transform_reduce(std::int64_t n0, std::int64_t n1, std::int64_t n2, T init, Map map, Combine combine) {
  return transform_reduce(
      detail::extent(detail::extent(n0, n1), n2), init, [=](std::int64_t i) { return map(i / (n1 * n2), i / n2 % n1, i % n2); }, combine);
}

template <class T, class Map, class Combine>
inline T transform_reduce(const Context &context, std::int64_t n0, std::int64_t n1, std::int64_t n2, T init, Map map, Combine combine) {
  return transform_reduce(
      context, detail::extent(detail::extent(n0, n1), n2), init, [=](std::int64_t i) { return map(i / (n1 * n2), i / n2 % n1, i % n2); },
      combine);
}

template <class Body> inline void task(Body body) {
  for_each(1, [=](std::int64_t) { return body(); });
}

template <class Body> inline void task(const Context &context, Body body) {
  for_each(context, 1, [=](std::int64_t) { return body(); });
}

} // namespace polyregion::offload
