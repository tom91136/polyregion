#pragma once

#include <cinttypes>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifdef __RT_IMPL
  #include <atomic>
  #include <chrono>
  #include <cstdio>
  #include <cstdlib>
  #include <mutex>
  #include <shared_mutex>

  #include "rt_hashmap.hpp"
  #include "rt_protected.hpp"

  #ifdef _WIN32
    #include <windows.h>
using pid_t = DWORD;
  #else
    #include <unistd.h>
using pid_t = pid_t;
  #endif

#endif

// NOLINTBEGIN(misc-definitions-in-headers)

namespace polyregion::rt_reflect {

extern "C" enum class Type : uint8_t {
  Unknown = 1,
  StackAlloc,
  StackFree,

  HeapMalloc,
  HeapCalloc,
  HeapRealloc,
  HeapMemalign,
  HeapAlignedAlloc,
  HeapFree,
  HeapCXXNew,
  HeapCXXDelete,
};

extern "C" struct PtrMeta {
  size_t size;
  size_t offset;
  Type type;
};

template <typename E> constexpr std::underlying_type_t<E> to_integral(E e) { return static_cast<std::underlying_type_t<E>>(e); }

constexpr const char *to_string(const Type t) {
  switch (t) {
    case Type::Unknown: return "Unknown";
    case Type::StackAlloc: return "StackAlloc";
    case Type::StackFree: return "StackFree";

    case Type::HeapMalloc: return "HeapMalloc";
    case Type::HeapCalloc: return "HeapCalloc";
    case Type::HeapRealloc: return "HeapRealloc";
    case Type::HeapMemalign: return "HeapMemalign";
    case Type::HeapAlignedAlloc: return "HeapAlignedAlloc";
    case Type::HeapFree: return "HeapFree";
    case Type::HeapCXXNew: return "HeapCXXNew";
    case Type::HeapCXXDelete: return "HeapCXXDelete";
  }
  return "Undefined";
}

#ifndef __RT_IMPL
extern "C" __attribute__((weak)) PtrMeta _rt_reflect_p(const void *ptr);
extern "C" __attribute__((weak)) PtrMeta _rt_reflect_v(uintptr_t ptrValue);
#endif

#ifdef __RT_IMPL

  #define __RT_EXPORTED __attribute__((noinline)) __attribute__((visibility("default")))

namespace details {

extern "C" struct PtrInfo {
  uintptr_t base;
  size_t size;
  Type type;
};

using namespace std::chrono;

__RT_PROTECT int safe_vasprintf(char **buffer, const char *format, va_list args) {
  va_list args0;
  va_copy(args0, args);
  int size = std::vsnprintf(nullptr, 0, format, args0);
  va_end(args0);

  if (size < 0) {
    va_end(args);
    *buffer = nullptr;
    return -1;
  }
  *buffer = static_cast<char *>(__RT_ALTERNATIVE(calloc)(1, size + 1));
  std::vsnprintf(*buffer, size + 1, format, args);
  va_end(args);
  return size;
}

__RT_PROTECT __attribute__((format(printf, 2, 3))) int safe_snprintf(char **buffer, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int size = safe_vasprintf(buffer, format, args);
  va_end(args);
  return size;
}

__RT_PROTECT __attribute__((format(printf, 2, 3))) int safe_fprintf(FILE *stream, const char *format, ...) {
  if (!stream) return 0;
  va_list args;
  va_start(args, format);
  char *buffer{};
  int size = safe_vasprintf(&buffer, format, args);
  va_end(args);
  if (size > 0) {
    auto written = static_cast<int>(fwrite(buffer, 1, size, stream));
    __RT_ALTERNATIVE(free)(buffer);
    return written;
  }
  return size;
}

__RT_PROTECT void fail() {
  std::fflush(stderr);
  std::abort();
}

struct PtrRecord {
  time_point<steady_clock> point;
  PtrInfo info;
};

class ReflectService {

  std::atomic_bool &interpose;
  HashMap<uintptr_t, PtrRecord> data;
  std::shared_mutex mutex{};
  time_point<steady_clock> start;
  std::FILE *trace{};

  __RT_PROTECT const PtrRecord *queryUnsafe(const uintptr_t ptr, const bool allowSubrange) const {
    const PtrRecord *result = data.find(ptr);
    if (result) return result;
    if (allowSubrange) {
      data.walk([&](uintptr_t, const PtrRecord *value) {
        if (ptr >= value->info.base && ptr < value->info.base + value->info.size) {
          result = value;
          return true;
        }
        return false;
      });
      return result;
    }
    return nullptr;
  }

public:
  __RT_PROTECT explicit ReflectService(std::atomic_bool &interpose)
      : interpose(interpose), data([](auto x) { return x; }), start(steady_clock::now()) {
    char *name{};
    safe_snprintf(&name, "trace_%d.json",
  #ifdef _WIN32
                  GetCurrentProcessId()
  #else
                  getpid()
  #endif
    );
    // trace = std::fopen(name, "w");
    __RT_ALTERNATIVE(free)(name);

    // safe_fprintf(trace, "[\n");
    safe_fprintf(stderr, "[PtrReflect] started\n");
    interpose = true;
  }

  __RT_PROTECT void blockingRecord(const PtrInfo &info, const time_point<steady_clock> now = steady_clock::now()) {
    std::unique_lock lock(mutex);
    // safe_fprintf(stderr, "[PtrReflect] record %p(size=%ld, type=%s)\n", reinterpret_cast<void *>(info.ptr), info.size,
    //              to_string(info.type));

    auto inserted = data.emplace(info.base, PtrRecord{now, info});
    if (!inserted) {
      safe_fprintf(stderr, "[PtrReflect] failed to insert %p (size=%ld, type=%s)\n", reinterpret_cast<void *>(info.base), info.size,
                   to_string(info.type));
      fail();
    }
  }

  __RT_PROTECT void blockingRelease(const uintptr_t ptr, const Type type, const time_point<steady_clock> now = steady_clock::now()) {
    std::unique_lock lock(mutex);
    // safe_fprintf(stderr, "[PtrReflect] release %p (type=%s)\n", reinterpret_cast<void *>(ptr), to_string(type));
    static int i = 0;
    if (const auto it = queryUnsafe(ptr, false)) {
      // const auto [recordPoint, info] = *it;
      // safe_fprintf(trace,
      //              "  {"
      //              "\"name\": \"0x%lx (%ld)\","
      //              "\"cat\": \"%s\", "
      //              "\"ph\": \"X\", "
      //              "\"ts\": %" PRId64 " , "
      //              "\"dur\": %" PRId64 ", \"pid\": %d, \"tid\": %d},\n",
      //              info.base, info.size, to_string(info.type),                                     //
      //              duration_cast<microseconds>(recordPoint.time_since_epoch()).count(),            //
      //              duration_cast<microseconds>(now - recordPoint).count(), to_integral(info.type), //
      //              0);
      // i++;
      // if (i % 100 == 0) std::fflush(trace);
      data.erase(ptr);
      return;
    }
    safe_fprintf(stderr, "[PtrReflect] failed to release %p (type=%s)\n", reinterpret_cast<void *>(ptr), to_string(type));
    // raise(SIGTRAP);
    // fail();
  }

  __RT_PROTECT PtrMeta blockingQuery(const uintptr_t ptrValue) {
    std::shared_lock lock(mutex);
    if (const auto it = queryUnsafe(ptrValue, true)) return PtrMeta{it->info.size, ptrValue - it->info.base, it->info.type};
    return PtrMeta{0, 0, Type::Unknown};
  }

  __RT_PROTECT PtrMeta blockingQuery(const void *ptr) { return blockingQuery(reinterpret_cast<uintptr_t>(ptr)); }

  __RT_PROTECT ~ReflectService() {
    interpose = false;
    // const auto now = steady_clock::now();
    // safe_fprintf(trace,
    //              "  {"
    //              "\"name\": \"runtime\","
    //              "\"cat\": \"global\", "
    //              "\"ph\": \"X\", "
    //              "\"ts\": %" PRId64 ", "
    //              "\"dur\": %" PRId64 ", \"pid\": 0, \"tid\": %d}\n",            //
    //              duration_cast<microseconds>(start.time_since_epoch()).count(), //
    //              duration_cast<microseconds>(now - start).count(),              //
    //              0);
    // safe_fprintf(trace, "]");
    // std::fclose(trace);
    safe_fprintf(stderr, "[PtrReflect] terminated\n");
  }
};

inline std::atomic_bool serviceInit{false};
__RT_PROTECT inline ReflectService *_rt_get() {
  static ReflectService s(serviceInit);
  return &s;
}
inline auto _ = _rt_get();

} // namespace details

extern "C" __RT_PROTECT __RT_EXPORTED void _rt_record(const void *ptr, const size_t size, const Type type) {
  if (!details::serviceInit.load()) return;
  details::_rt_get()->blockingRecord(details::PtrInfo{reinterpret_cast<uintptr_t>(ptr), size, type});
}
extern "C" __RT_PROTECT __RT_EXPORTED void _rt_release(void *ptr, const Type type) {
  if (!ptr) return;
  if (!details::serviceInit.load()) return;
  details::_rt_get()->blockingRelease(reinterpret_cast<uintptr_t>(ptr), type);
}

extern "C" __RT_PROTECT __RT_EXPORTED PtrMeta _rt_reflect_p(const void *ptr) { return details::_rt_get()->blockingQuery(ptr); }
extern "C" __RT_PROTECT __RT_EXPORTED PtrMeta _rt_reflect_v(const uintptr_t ptrValue) {
  return details::_rt_get()->blockingQuery(ptrValue);
}

#endif
} // namespace polyregion::rt_reflect

// NOLINTEND(misc-definitions-in-headers)