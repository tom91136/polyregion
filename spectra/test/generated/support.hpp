#pragma once

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Host-oracle comparison helpers shared by the generated Spectra API tests.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

#if defined(SPECTRA_NATIVE_THRUST) && defined(__HIPCC__)
  #include <hip/hip_runtime_api.h>
  #define SPECTRA_TEST_FN __host__ __device__
#elif defined(SPECTRA_NATIVE_THRUST) && defined(__CUDACC__)
  #include <cuda_runtime_api.h>
  #define SPECTRA_TEST_FN __host__ __device__
#elif defined(SPECTRA_NATIVE_ONEDPL)
  #include <oneapi/dpl/execution>
  #include <sycl/sycl.hpp>
  #define SPECTRA_TEST_FN
#else
  #define SPECTRA_TEST_FN
#endif

using spectra_test_value4 = std::int32_t;
using spectra_test_value8 = std::int64_t;

template <class T> struct spectra_test_allocator {
  using value_type = T;

  spectra_test_allocator() noexcept = default;
  template <class U> spectra_test_allocator(const spectra_test_allocator<U> &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t n) {
#if defined(SPECTRA_NATIVE_THRUST) && defined(__HIPCC__)
    void *ptr = nullptr;
    if (hipMallocManaged(&ptr, n * sizeof(T)) != hipSuccess) throw std::bad_alloc();
    return static_cast<T *>(ptr);
#elif defined(SPECTRA_NATIVE_THRUST) && defined(__CUDACC__)
    void *ptr = nullptr;
    if (cudaMallocManaged(&ptr, n * sizeof(T)) != cudaSuccess) throw std::bad_alloc();
    return static_cast<T *>(ptr);
#elif defined(SPECTRA_NATIVE_ONEDPL)
    T *ptr = sycl::malloc_shared<T>(n, oneapi::dpl::execution::dpcpp_default.queue());
    if (!ptr) throw std::bad_alloc();
    return ptr;
#else
    return std::allocator<T>{}.allocate(n);
#endif
  }

  void deallocate(T *ptr, std::size_t n) noexcept {
#if defined(SPECTRA_NATIVE_THRUST) && defined(__HIPCC__)
    (void)n;
    (void)hipFree(ptr);
#elif defined(SPECTRA_NATIVE_THRUST) && defined(__CUDACC__)
    (void)n;
    (void)cudaFree(ptr);
#elif defined(SPECTRA_NATIVE_ONEDPL)
    (void)n;
    sycl::free(ptr, oneapi::dpl::execution::dpcpp_default.queue());
#else
    std::allocator<T>{}.deallocate(ptr, n);
#endif
  }

  template <class U> bool operator==(const spectra_test_allocator<U> &) const noexcept { return true; }
  template <class U> bool operator!=(const spectra_test_allocator<U> &) const noexcept { return false; }
};

template <class T> using spectra_test_vector = std::vector<T, spectra_test_allocator<T>>;

struct spectra_test_value12 {
  std::int32_t x, y, z;
  SPECTRA_TEST_FN constexpr spectra_test_value12(std::int32_t value = 0) : x(value), y(value), z(value) {}
  SPECTRA_TEST_FN constexpr spectra_test_value12(std::int32_t x_, std::int32_t y_, std::int32_t z_) : x(x_), y(y_), z(z_) {}
  template <class I, typename std::enable_if<std::is_integral<I>::value, int>::type = 0>
  SPECTRA_TEST_FN constexpr explicit operator I() const {
    return static_cast<I>(x);
  }
};
static_assert(sizeof(spectra_test_value4) == 4);
static_assert(sizeof(spectra_test_value8) == 8);
static_assert(sizeof(spectra_test_value12) == 12);
static_assert(std::is_trivially_copyable_v<spectra_test_value12>);

SPECTRA_TEST_FN constexpr spectra_test_value12 operator+(spectra_test_value12 a, spectra_test_value12 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
SPECTRA_TEST_FN constexpr spectra_test_value12 operator-(spectra_test_value12 a, spectra_test_value12 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
SPECTRA_TEST_FN constexpr spectra_test_value12 operator*(spectra_test_value12 a, spectra_test_value12 b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
SPECTRA_TEST_FN constexpr bool operator==(spectra_test_value12 a, spectra_test_value12 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
SPECTRA_TEST_FN constexpr bool operator!=(spectra_test_value12 a, spectra_test_value12 b) { return !(a == b); }
SPECTRA_TEST_FN constexpr bool operator<(spectra_test_value12 a, spectra_test_value12 b) {
  return a.x != b.x ? a.x < b.x : a.y != b.y ? a.y < b.y : a.z < b.z;
}
SPECTRA_TEST_FN constexpr bool operator>(spectra_test_value12 a, spectra_test_value12 b) { return b < a; }
SPECTRA_TEST_FN constexpr bool operator<=(spectra_test_value12 a, spectra_test_value12 b) { return !(b < a); }
SPECTRA_TEST_FN constexpr bool operator>=(spectra_test_value12 a, spectra_test_value12 b) { return !(a < b); }

template <class T> static bool near_(T a, T b) {
  if constexpr (std::is_floating_point<T>::value) {
    double d = a > b ? double(a) - double(b) : double(b) - double(a);
    double m = double(a > 0 ? a : -a), mb = double(b > 0 ? b : -b), s = m > mb ? m : mb;
    return d <= 1e-6 * (s + 1.0);
  } else return a == b;
}
template <class A, class B> static bool vecNear(const A &a, const B &b, int n) {
  for (int i = 0; i < n; ++i)
    if (!near_(a[i], b[i])) return false;
  return true;
}
template <class A, class B> static bool sameSeq(const A &a, const B &b, int n) {
  for (int i = 0; i < n; ++i)
    if (a[i] != b[i]) return false;
  return true;
}
