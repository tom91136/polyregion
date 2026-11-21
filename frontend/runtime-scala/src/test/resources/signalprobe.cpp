#include <atomic>
#include <cstdint>
#include <string>

#include "polyrt/page_guard.hpp"

#include "jni_utils.h"
#include <sys/mman.h>

namespace guard = polyregion::polyrt::guard;

namespace {

constexpr const char *EX = "java/lang/RuntimeException";

std::atomic<jlong> resolutions{0};
std::atomic<uintptr_t> guardedBegin{0};
std::atomic<size_t> guardedBytes{0};

guard::Resolution resolve(void *address) {
  const auto begin = guardedBegin.load();
  if (reinterpret_cast<uintptr_t>(address) < begin || reinterpret_cast<uintptr_t>(address) >= begin + guardedBytes.load())
    return guard::Resolution::Declined;
  guard::protect(reinterpret_cast<void *>(begin), guardedBytes.load(), guard::Access::ReadWrite);
  ++resolutions;
  return guard::Resolution::Resolved;
}

void foreignHandler(int signal, siginfo_t *, void *) {
  struct sigaction fallback{};
  fallback.sa_handler = SIG_DFL;
  sigemptyset(&fallback.sa_mask);
  sigaction(signal, &fallback, nullptr);
}

void installForeign() {
  struct sigaction action{};
  action.sa_sigaction = foreignHandler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  sigaction(SIGSEGV, &action, nullptr);
}

void armGuard(const jlong base, const jlong bytes) {
  guardedBegin = static_cast<uintptr_t>(base);
  guardedBytes = static_cast<size_t>(bytes);
  guard::track(static_cast<uintptr_t>(base), static_cast<uintptr_t>(base + bytes));
  guard::install(resolve);
  guard::protect(reinterpret_cast<void *>(base), static_cast<size_t>(bytes), guard::Access::Read);
}

struct sigaction snapshotted[SignalGuard::COUNT]{};

constexpr int KernelAddedFlags = 0x04000000; // SA_RESTORER, set by libc on any sigaction call

bool sameAction(const struct sigaction &a, const struct sigaction &b) {
  return a.sa_handler == b.sa_handler && (a.sa_flags & ~KernelAddedFlags) == (b.sa_flags & ~KernelAddedFlags);
}

template <class Expected> std::string differing(Expected &&expected) {
  std::string names;
  for (size_t i = 0; i < SignalGuard::COUNT; ++i) {
    struct sigaction current{};
    sigaction(SignalGuard::SIGNALS[i], nullptr, &current);
    if (!expected(i, current)) names += (names.empty() ? "" : ",") + std::to_string(SignalGuard::SIGNALS[i]);
  }
  return names;
}

} // namespace

extern "C" {

JNIEXPORT void JNICALL Java_polyregion_SignalNatives_snapshot(JNIEnv *, jclass) {
  for (size_t i = 0; i < SignalGuard::COUNT; ++i)
    sigaction(SignalGuard::SIGNALS[i], nullptr, &snapshotted[i]);
}

JNIEXPORT jstring JNICALL Java_polyregion_SignalNatives_changed(JNIEnv *env, jclass) {
  return env->NewStringUTF(
      differing([](const size_t i, const struct sigaction &current) { return sameAction(current, snapshotted[i]); }).c_str());
}

JNIEXPORT jstring JNICALL Java_polyregion_SignalNatives_nestedGuards(JNIEnv *env, jclass) {
  Java_polyregion_SignalNatives_snapshot(env, nullptr);
  std::string during;
  wrapBackend(env, EX, [&] {
    for (const auto signal : SignalGuard::SIGNALS) {
      struct sigaction action{};
      action.sa_sigaction = foreignHandler;
      action.sa_flags = SA_SIGINFO;
      sigemptyset(&action.sa_mask);
      sigaction(signal, &action, nullptr);
    }
    {
      const SignalGuard inner;
    }
    during = differing([](const size_t, const struct sigaction &current) { return current.sa_sigaction == foreignHandler; });
  });
  const auto after = differing([](const size_t i, const struct sigaction &current) { return sameAction(current, snapshotted[i]); });
  return env->NewStringUTF(("during=" + during + ";after=" + after).c_str());
}

JNIEXPORT jlong JNICALL Java_polyregion_SignalNatives_map(JNIEnv *, jclass, jlong bytes) {
  void *p = mmap(nullptr, static_cast<size_t>(bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return p == MAP_FAILED ? 0 : static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
}

JNIEXPORT jobject JNICALL Java_polyregion_SignalNatives_buffer(JNIEnv *env, jclass, jlong base, jlong bytes) {
  return env->NewDirectByteBuffer(reinterpret_cast<void *>(base), bytes);
}

JNIEXPORT void JNICALL Java_polyregion_SignalNatives_guard(JNIEnv *env, jclass, jlong base, jlong bytes) {
  wrapBackend(env, EX, [&] { armGuard(base, bytes); });
}

JNIEXPORT void JNICALL Java_polyregion_SignalNatives_foreignThenGuard(JNIEnv *env, jclass, jlong base, jlong bytes) {
  wrapBackend(env, EX, [&] {
    installForeign();
    armGuard(base, bytes);
  });
}

JNIEXPORT void JNICALL Java_polyregion_SignalNatives_foreignThenInstall(JNIEnv *env, jclass) {
  wrapBackend(env, EX, [&] {
    installForeign();
    guard::install(resolve);
  });
}

JNIEXPORT void JNICALL Java_polyregion_SignalNatives_guardThenForeign(JNIEnv *env, jclass, jlong base, jlong bytes) {
  wrapBackend(env, EX, [&] {
    armGuard(base, bytes);
    installForeign();
  });
}

JNIEXPORT void JNICALL Java_polyregion_SignalNatives_foreign(JNIEnv *env, jclass) {
  wrapBackend(env, EX, [&] { installForeign(); });
}

JNIEXPORT void JNICALL Java_polyregion_SignalNatives_rearm(JNIEnv *, jclass, jlong base, jlong bytes) {
  guard::protect(reinterpret_cast<void *>(base), static_cast<size_t>(bytes), guard::Access::Read);
}

JNIEXPORT jlong JNICALL Java_polyregion_SignalNatives_resolutions(JNIEnv *, jclass) { return resolutions.load(); }
}
