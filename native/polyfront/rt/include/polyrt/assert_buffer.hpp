#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "polyregion/conventions.h"
#include "polyrt/rt.h"

namespace polyregion::polyrt {

inline constexpr size_t assertCodeBytes = sizeof(uint32_t); // LE prefix before the message
inline constexpr size_t assertBufferBytes = assertCodeBytes + conventions::AssertMessageLimit;

struct AssertReason {
  bool raised;
  uint32_t code;
  std::string message;
  bool truncated;
};

class AssertSink {
  mutable std::mutex mutex;
  AssertReason last{};
  std::atomic<bool> flag{false};

public:
  void reset() {
    flag.store(false, std::memory_order_relaxed);
    const std::lock_guard guard(mutex);
    last = {};
  }
  void record(const AssertReason &reason) {
    if (!reason.raised) return;
    const std::lock_guard guard(mutex);
    last = reason;
    flag.store(true, std::memory_order_relaxed);
  }
  void record(const uint32_t code, const char *message) { record({true, code, message ? std::string(message) : std::string(), false}); }
  bool raised() const { return flag.load(std::memory_order_relaxed); }
  AssertReason reason() const {
    const std::lock_guard guard(mutex);
    return last;
  }
};

inline AssertReason decodeAssertBuffer(const uint8_t *buf) {
  const uint32_t code = uint32_t(buf[0]) | (uint32_t(buf[1]) << 8) | (uint32_t(buf[2]) << 16) | (uint32_t(buf[3]) << 24);
  const auto *msg = reinterpret_cast<const char *>(buf + assertCodeBytes);
  const size_t len = ::strnlen(msg, conventions::AssertMessageLimit);
  return {code != 0 || len != 0, code, std::string(msg, len), len == conventions::AssertMessageLimit};
}

inline void logAssertReason(const AssertReason &r, const char *fn, const char *moduleId) {
  if (r.raised)
    log(DebugLevel::Info, "<%s:%s> kernel asserted: code=%u message=\"%s\"%s", fn, moduleId, r.code, r.message.c_str(),
        r.truncated ? " (truncated)" : "");
}

inline void reportAssert(const AssertReason &reason, AssertSink &sink, const char *fn, const char *moduleId) {
  logAssertReason(reason, fn, moduleId);
  sink.record(reason);
}

inline void bindAssertError(invoke::ArgBuffer &buffer, const bool asserts, uintptr_t &errDev) {
  if (asserts) buffer.append(invoke::Type::Ptr, &errDev);
}

inline void appendArgTerminator(invoke::ArgBuffer &buffer) { buffer.append(invoke::Type::Void, nullptr); }

inline uintptr_t allocAssertBuffer() {
  const auto errDev = currentDevice->mallocDevice(assertBufferBytes, Access::RW);
  const std::vector<uint8_t> zero(assertBufferBytes, 0);
  currentQueue->enqueueHostToDeviceAsyncTyped(zero.data(), errDev, assertBufferBytes, {});
  currentQueue->enqueueWaitBlocking();
  return errDev;
}

inline AssertReason readAssertBuffer(const uintptr_t errDev) {
  std::vector<uint8_t> buf(assertBufferBytes, 0);
  currentQueue->enqueueDeviceToHostAsyncTyped(errDev, buf.data(), assertBufferBytes, {});
  currentQueue->enqueueWaitBlocking();
  currentDevice->freeDevice(errDev);
  return decodeAssertBuffer(buf.data());
}

} // namespace polyregion::polyrt
