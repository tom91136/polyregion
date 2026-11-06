#pragma region case: msvc-wide-assert
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 ASRT assert_platform.cpp:7: Assertion `v >= 0' failed.

#pragma region case: msvc-narrow-assert
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 ASRT assert_platform.cpp:7: Assertion `v >= 0' failed.

#pragma region case: bsd-assert-rtn
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 ASRT assert_platform.cpp:7: kernel: Assertion `v >= 0' failed.

#pragma region case: system-assert
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 ASRT yes

#pragma region case: other-noreturn-sink
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 ERRO abort

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "polyregion/enums.h"

#include "test_utils.h"

#if CHECK_KIND == 0
  #if !defined(_WIN32)
extern "C" void _wassert(const wchar_t *message, const wchar_t *file, unsigned line);
  #endif
  #define PLATFORM_ASSERT(e) ((e) ? void(0) : _wassert(L"" #e, L"assert_platform.cpp", 7u))
#elif CHECK_KIND == 1
extern "C" void _assert(const char *message, const char *file, unsigned line);
  #define PLATFORM_ASSERT(e) ((e) ? void(0) : _assert(#e, "assert_platform.cpp", 7u))
#elif CHECK_KIND == 2
  #if !defined(__APPLE__)
extern "C" [[noreturn]] void __assert_rtn(const char *function, const char *file, int line, const char *expression);
  #endif
  #define PLATFORM_ASSERT(e) ((e) ? void(0) : __assert_rtn("kernel", "assert_platform.cpp", 7, #e))
#elif CHECK_KIND == 3
  #define PLATFORM_ASSERT(e) assert(e)
#elif CHECK_KIND == 4
  #define PLATFORM_ASSERT(e) ((e) ? void(0) : std::abort())
#else
  #error "CHECK_KIND undefined"
#endif

int main() {
  int value = -1;
  int *p = &value;
  __polyregion_offload_f1__([=]() {
    const int v = *p;
    PLATFORM_ASSERT(v >= 0);
    return 0;
  });
  const auto a = polyregion::polystl::details::lastAssert();
  const char cc[5] = {char(a.code), char(a.code >> 8), char(a.code >> 16), char(a.code >> 24), 0};
#if CHECK_KIND == 3
  const bool named = a.message.find("Assertion `v >= 0' failed.") != std::string::npos;
  std::printf("%d %s %s", a.raised ? 1 : 0, cc, named ? "yes" : a.message.c_str());
#else
  std::printf("%d %s %s", a.raised ? 1 : 0, cc, a.message.c_str());
#endif
  return 0;
}

#if CHECK_KIND == 0 && !defined(_WIN32)
extern "C" void _wassert(const wchar_t *, const wchar_t *, unsigned) { __builtin_trap(); }
#elif CHECK_KIND == 1 && !defined(_WIN32)
extern "C" void _assert(const char *, const char *, unsigned) { __builtin_trap(); }
#elif CHECK_KIND == 2 && !defined(__APPLE__)
extern "C" [[noreturn]] void __assert_rtn(const char *, const char *, int, const char *) { __builtin_trap(); }
#endif
