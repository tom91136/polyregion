#pragma once

#include <cstddef>
#include <cstdlib>

#if defined(__linux__) || defined(__APPLE__)
  #include <dlfcn.h>
#endif

#if defined(__clang__)
  #define __RT_ANNOTATE(name) [[clang::annotate(name)]]
#else
  #define __RT_ANNOTATE(name)
#endif

#if defined(_MSC_VER)
  #define __RT_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
  #define __RT_NOINLINE __attribute__((noinline))
#else
  #define __RT_NOINLINE
#endif

#define __RT_ODR __RT_ANNOTATE("polyreflect-rt-odr")

#if defined(__linux__) || defined(__APPLE__)

  #define __DEF_DLSYM(func, ret_type, ...) ret_type (*_##func)(__VA_ARGS__) = (ret_type (*)(__VA_ARGS__))dlsym(RTLD_NEXT, #func)

extern "C" inline void *__interposed_malloc(const size_t size) {
  static __DEF_DLSYM(malloc, void *, size_t);
  return _malloc(size);
}

extern "C" inline void *__interposed_calloc(const size_t nmemb, const size_t size) {
  static __DEF_DLSYM(calloc, void *, size_t, size_t);
  return _calloc(nmemb, size);
}

extern "C" inline void *__interposed_memalign(const size_t alignment, const size_t size) {
  static __DEF_DLSYM(memalign, void *, size_t, size_t);
  return _memalign(alignment, size);
}

extern "C" inline void *__interposed_realloc(void *ptr, const size_t size) {
  static __DEF_DLSYM(realloc, void *, void *, size_t);
  return _realloc(ptr, size);
}

extern "C" inline void __interposed_free(void *ptr) {
  static __DEF_DLSYM(free, void, void *);
  _free(ptr);
}

#else

// XXX HeapAlloc bypasses InterposePass; std::malloc would recurse via _rt_record during init.
  #include <windows.h>
extern "C" inline void *__interposed_malloc(const size_t size) { return ::HeapAlloc(::GetProcessHeap(), 0, size); }
extern "C" inline void *__interposed_calloc(const size_t nmemb, const size_t size) {
  return ::HeapAlloc(::GetProcessHeap(), HEAP_ZERO_MEMORY, nmemb * size);
}
extern "C" inline void *__interposed_realloc(void *ptr, const size_t size) {
  if (!ptr) return ::HeapAlloc(::GetProcessHeap(), 0, size);
  return ::HeapReAlloc(::GetProcessHeap(), 0, ptr, size);
}
extern "C" inline void *__interposed_memalign(size_t, const size_t size) { return ::HeapAlloc(::GetProcessHeap(), 0, size); }
extern "C" inline void __interposed_free(void *ptr) {
  if (ptr) ::HeapFree(::GetProcessHeap(), 0, ptr);
}

#endif

#if defined(_MSC_VER)
  #define __RT_ALTERNATIVE(func) __interposed_##func
#else
extern "C" __attribute__((weak)) void *__interceptor_malloc(size_t size);
extern "C" __attribute__((weak)) void *__interceptor_calloc(size_t nmemb, size_t size);
extern "C" __attribute__((weak)) void *__interceptor_realloc(void *ptr, size_t size);
extern "C" __attribute__((weak)) void *__interceptor_memalign(size_t alignment, size_t size);
extern "C" __attribute__((weak)) void __interceptor_free(void *ptr);

  #define __RT_ALTERNATIVE(func) (__interceptor_##func ? __interceptor_##func : __interposed_##func)
#endif
