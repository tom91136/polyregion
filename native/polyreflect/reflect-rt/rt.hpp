#pragma once

#include "rt_protected.hpp"
#include "rt_reflect.hpp"

#if !defined(__RT_IMPL) && !defined(__RT_NO_GLOBAL_NEW)

  #include <cstddef>

extern "C" void *polyrt_record_operator_new(std::size_t);
extern "C" void polyrt_record_operator_delete(void *);
extern "C" void polyrt_record_operator_delete_sized(void *, std::size_t);

// NOLINTBEGIN(misc-definitions-in-headers): ProtectRT marks these LinkOnceODR per-TU
__RT_ODR void *operator new(std::size_t n) { return polyrt_record_operator_new(n); }
__RT_ODR void *operator new[](std::size_t n) { return polyrt_record_operator_new(n); }
__RT_ODR void operator delete(void *p) noexcept { polyrt_record_operator_delete(p); }
__RT_ODR void operator delete[](void *p) noexcept { polyrt_record_operator_delete(p); }
__RT_ODR void operator delete(void *p, std::size_t n) noexcept { polyrt_record_operator_delete_sized(p, n); }
__RT_ODR void operator delete[](void *p, std::size_t n) noexcept { polyrt_record_operator_delete_sized(p, n); }
// NOLINTEND(misc-definitions-in-headers)

#endif
