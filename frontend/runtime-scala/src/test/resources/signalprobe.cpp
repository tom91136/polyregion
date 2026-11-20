#include <atomic>
#include <cstdint>

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

} // namespace

extern "C" {

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
