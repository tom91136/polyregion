#pragma once

#include <cstddef>
#include <new>

#include "rt_protected.hpp"
#include "rt_reflect.hpp"

#define __ALLOC __RT_PROTECT [[clang::annotate("__rt_alloc")]] __attribute__((noinline))
#define __FREE __RT_PROTECT [[clang::annotate("__rt_free")]]

#if __cpp_exceptions == 199711
  #define __THROW_OF_ABORT(e) throw e
#else
  #define __THROW_OF_ABORT(e) std::abort()
#endif

// NOLINTBEGIN(misc-definitions-in-headers)

extern "C" __ALLOC void *malloc(const size_t size) {
  const auto ptr = __RT_ALTERNATIVE(malloc)(size);
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapMalloc);
  return ptr;
}

extern "C" __ALLOC void *calloc(const size_t nmemb, const size_t size) {
  const auto ptr = __RT_ALTERNATIVE(calloc)(nmemb, size);
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapCalloc);
  return ptr;
}

extern "C" __ALLOC void *realloc(void *ptr, const size_t size) {
  if (ptr) _rt_release(ptr, polyregion::rt_reflect::Type::HeapFree);
  const auto ptr1 = __RT_ALTERNATIVE(realloc)(ptr, size);
  _rt_record(ptr1, size, polyregion::rt_reflect::Type::HeapRealloc);
  return ptr1;
}

extern "C" __ALLOC void *memalign(const size_t alignment, const size_t size) {
  const auto ptr = __RT_ALTERNATIVE(memalign)(alignment, size);
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapMemalign);
  return ptr;
}

extern "C" __ALLOC void *aligned_alloc(size_t alignment, size_t size) {
  const auto ptr = __RT_ALTERNATIVE(memalign)(alignment, size);
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapAlignedAlloc);
  return ptr;
}

extern "C" __FREE void free(void *ptr) {
  __RT_ALTERNATIVE(free)(ptr);
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapFree);
}

__ALLOC void *operator new(const size_t size) {
  auto *ptr = __RT_ALTERNATIVE(malloc)(size);
  if (!ptr) __THROW_OF_ABORT(std::bad_alloc{});
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapCXXNew);
  return ptr;
}

__ALLOC void *operator new(const size_t size, std::align_val_t a) {
  auto *ptr = __RT_ALTERNATIVE(memalign)(static_cast<size_t>(a), size);
  if (!ptr) __THROW_OF_ABORT(std::bad_alloc{});
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapCXXNew);
  return ptr;
}

__ALLOC void *operator new(const size_t size, const std::nothrow_t &) noexcept {
  const auto ptr = __RT_ALTERNATIVE(malloc)(size);
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapCXXNew);
  return ptr;
}

__ALLOC void *operator new(const size_t size, std::align_val_t a, const std::nothrow_t &) noexcept {
  const auto ptr = __RT_ALTERNATIVE(malloc)(size);
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapCXXNew);
  return ptr;
}

__ALLOC void *operator new[](const size_t size) {
  auto *ptr = __RT_ALTERNATIVE(malloc)(size);
  if (!ptr) __THROW_OF_ABORT(std::bad_alloc{});
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapCXXNew);
  return ptr;
}

__ALLOC void *operator new[](const size_t size, std::align_val_t a) {
  auto *ptr = __RT_ALTERNATIVE(memalign)(static_cast<size_t>(a), size);
  if (!ptr) __THROW_OF_ABORT(std::bad_alloc{});
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapCXXNew);
  return ptr;
}

__ALLOC void *operator new[](const size_t size, const std::nothrow_t &) noexcept {
  const auto ptr = __RT_ALTERNATIVE(malloc)(size);
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapCXXNew);
  return ptr;
}

__ALLOC void *operator new[](const size_t size, std::align_val_t a, const std::nothrow_t &) noexcept {
  const auto ptr = __RT_ALTERNATIVE(memalign)(static_cast<size_t>(a), size);
  _rt_record(ptr, size, polyregion::rt_reflect::Type::HeapCXXNew);
  return ptr;
}

__FREE void operator delete(void *ptr) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete[](void *ptr) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete(void *ptr, std::align_val_t) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete[](void *ptr, std::align_val_t) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete(void *ptr, size_t) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete[](void *ptr, size_t) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete(void *ptr, size_t, std::align_val_t) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete[](void *ptr, size_t, std::align_val_t) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete(void *ptr, const std::nothrow_t &) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete[](void *ptr, const std::nothrow_t &) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete(void *ptr, std::align_val_t, const std::nothrow_t &) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}
__FREE void operator delete[](void *ptr, std::align_val_t, const std::nothrow_t &) noexcept {
  _rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}

// NOLINTEND(misc-definitions-in-headers)