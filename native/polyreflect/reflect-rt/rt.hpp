#pragma once

#include "rt_protected.hpp"
#include "rt_reflect.hpp"

#if !defined(__RT_IMPL) && !defined(__RT_NO_GLOBAL_NEW)

  #include <cstddef>
  #include <new>

// NOLINTNEXTLINE(readability-redundant-declaration)
void *polyrt_record_operator_new(std::size_t);
extern "C" void *polyrt_record_operator_new_nothrow(std::size_t, const void *);
extern "C" void *polyrt_record_operator_new_aligned(std::size_t, std::size_t);
extern "C" void *polyrt_record_operator_new_aligned_nothrow(std::size_t, std::size_t, const void *);
// NOLINTNEXTLINE(readability-redundant-declaration)
extern "C" void polyrt_record_operator_delete(void *);
// NOLINTNEXTLINE(readability-redundant-declaration)
extern "C" void polyrt_record_operator_delete_sized(void *, std::size_t);
extern "C" void polyrt_record_operator_delete_nothrow(void *, const void *);
extern "C" void polyrt_record_operator_delete_aligned(void *, std::size_t);
extern "C" void polyrt_record_operator_delete_aligned_sized(void *, std::size_t, std::size_t);
extern "C" void polyrt_record_operator_delete_aligned_nothrow(void *, std::size_t, const void *);

// NOLINTBEGIN(misc-definitions-in-headers): ProtectRT marks these LinkOnceODR per-TU
__RT_ODR void *operator new(std::size_t n) { return polyrt_record_operator_new(n); }
__RT_ODR void *operator new[](std::size_t n) { return polyrt_record_operator_new(n); }
__RT_ODR void *operator new(std::size_t n, const std::nothrow_t &t) noexcept { return polyrt_record_operator_new_nothrow(n, &t); }
__RT_ODR void *operator new[](std::size_t n, const std::nothrow_t &t) noexcept { return polyrt_record_operator_new_nothrow(n, &t); }
__RT_ODR void *operator new(std::size_t n, std::align_val_t a) {
  return polyrt_record_operator_new_aligned(n, static_cast<std::size_t>(a));
}
__RT_ODR void *operator new[](std::size_t n, std::align_val_t a) {
  return polyrt_record_operator_new_aligned(n, static_cast<std::size_t>(a));
}
__RT_ODR void *operator new(std::size_t n, std::align_val_t a, const std::nothrow_t &t) noexcept {
  return polyrt_record_operator_new_aligned_nothrow(n, static_cast<std::size_t>(a), &t);
}
__RT_ODR void *operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t &t) noexcept {
  return polyrt_record_operator_new_aligned_nothrow(n, static_cast<std::size_t>(a), &t);
}
__RT_ODR void operator delete(void *p) noexcept { polyrt_record_operator_delete(p); }
__RT_ODR void operator delete[](void *p) noexcept { polyrt_record_operator_delete(p); }
__RT_ODR void operator delete(void *p, std::size_t n) noexcept { polyrt_record_operator_delete_sized(p, n); }
__RT_ODR void operator delete[](void *p, std::size_t n) noexcept { polyrt_record_operator_delete_sized(p, n); }
__RT_ODR void operator delete(void *p, const std::nothrow_t &t) noexcept { polyrt_record_operator_delete_nothrow(p, &t); }
__RT_ODR void operator delete[](void *p, const std::nothrow_t &t) noexcept { polyrt_record_operator_delete_nothrow(p, &t); }
__RT_ODR void operator delete(void *p, std::align_val_t a) noexcept {
  polyrt_record_operator_delete_aligned(p, static_cast<std::size_t>(a));
}
__RT_ODR void operator delete[](void *p, std::align_val_t a) noexcept {
  polyrt_record_operator_delete_aligned(p, static_cast<std::size_t>(a));
}
__RT_ODR void operator delete(void *p, std::size_t n, std::align_val_t a) noexcept {
  polyrt_record_operator_delete_aligned_sized(p, n, static_cast<std::size_t>(a));
}
__RT_ODR void operator delete[](void *p, std::size_t n, std::align_val_t a) noexcept {
  polyrt_record_operator_delete_aligned_sized(p, n, static_cast<std::size_t>(a));
}
__RT_ODR void operator delete(void *p, std::align_val_t a, const std::nothrow_t &t) noexcept {
  polyrt_record_operator_delete_aligned_nothrow(p, static_cast<std::size_t>(a), &t);
}
__RT_ODR void operator delete[](void *p, std::align_val_t a, const std::nothrow_t &t) noexcept {
  polyrt_record_operator_delete_aligned_nothrow(p, static_cast<std::size_t>(a), &t);
}
// NOLINTEND(misc-definitions-in-headers)

#endif
