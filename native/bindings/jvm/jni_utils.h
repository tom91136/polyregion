#pragma once
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef _WIN32
  #include <csignal>
  #include <cstddef>
  #include <mutex>
#endif

#include "jni.h"

static JNIEnv *getEnv(JavaVM *vm) {
  JNIEnv *env{};
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_1) != JNI_OK) {
    return nullptr;
  }
  return env;
}

static jstring toJni(JNIEnv *env, const std::string &s) { return env->NewStringUTF(s.c_str()); }

template <typename C, typename F> static jobjectArray toJni(JNIEnv *env, C &xs, jclass clazz, F &&f) {
  // XXX xs is non-const because it may hold things that we std::move from
  auto *ys = env->NewObjectArray(jsize(xs.size()), clazz, nullptr);
  for (jsize i = 0; i < jsize(xs.size()); ++i) {
    jobject local = f(xs[i]);
    env->SetObjectArrayElement(ys, i, local);
    env->DeleteLocalRef(local);
  }
  return ys;
}

static std::string fromJni(JNIEnv *env, const jstring &s) {
  const auto *data = env->GetStringUTFChars(s, nullptr);
  std::string out(data);
  env->ReleaseStringUTFChars(s, data);
  return out;
}
template <typename T>
static std::vector<T> fromJni(JNIEnv *env, jbyteArray xs)
  requires(std::is_convertible_v<T, char>)
{
  auto *xsData = env->GetByteArrayElements(xs, nullptr);
  auto bytes = std::vector<T>(xsData, xsData + env->GetArrayLength(xs));
  env->ReleaseByteArrayElements(xs, xsData, JNI_ABORT);
  return bytes;
}

static std::vector<jlong> fromJni(JNIEnv *env, jlongArray xs) {
  std::vector<jlong> ys(env->GetArrayLength(xs));
  env->GetLongArrayRegion(xs, 0, jsize(ys.size()), ys.data());
  return ys;
}

template <typename T = std::nullptr_t> static T throwGeneric(JNIEnv *env, const std::string &exceptionClass, const std::string &message) {
  if (auto *exClass = env->FindClass(exceptionClass.c_str()); exClass) {
    if (env->ThrowNew(exClass, message.c_str()) != JNI_OK) {
      throw std::logic_error("Cannot throw exception of class " + exceptionClass + ", message was: " + message);
    } else {
      return T();
    }
  } else {
    throw std::logic_error("Cannot throw exception of unknown class " + exceptionClass + ", message was: " + message);
  }
}

template <typename F> static auto wrapException(JNIEnv *env, const std::string &exceptionClass, F &&f) {
  try {
    return f();
  } catch (const std::exception &e) {
    return throwGeneric<std::invoke_result_t<F>>(env, exceptionClass, e.what());
  }
}

#ifndef _WIN32
class SignalGuard {
  static constexpr int SIGNALS[] = {SIGHUP,  SIGINT,  SIGQUIT, SIGILL,  SIGTRAP, SIGABRT, SIGBUS, SIGFPE,
                                    SIGSEGV, SIGPIPE, SIGUSR2, SIGTERM, SIGXCPU, SIGXFSZ, SIGSYS};
  static constexpr size_t COUNT = sizeof(SIGNALS) / sizeof(SIGNALS[0]);

  struct State {
    std::mutex mutex;
    size_t active = 0;
    struct sigaction trusted[COUNT]{};
  };

  static State &state() {
    static State s;
    return s;
  }

public:
  SignalGuard() {
    auto &s = state();
    const std::scoped_lock lock(s.mutex);
    if (s.active++ == 0)
      for (size_t i = 0; i < COUNT; ++i)
        sigaction(SIGNALS[i], nullptr, &s.trusted[i]);
  }
  SignalGuard(const SignalGuard &) = delete;
  SignalGuard &operator=(const SignalGuard &) = delete;
  ~SignalGuard() {
    auto &s = state();
    const std::scoped_lock lock(s.mutex);
    for (size_t i = 0; i < COUNT; ++i) {
      struct sigaction current{};
      if (sigaction(SIGNALS[i], nullptr, &current) == 0
          && (current.sa_handler != s.trusted[i].sa_handler || current.sa_flags != s.trusted[i].sa_flags))
        sigaction(SIGNALS[i], &s.trusted[i], nullptr);
    }
    --s.active;
  }
};
#else
struct SignalGuard {};
#endif

template <typename F> static auto wrapBackend(JNIEnv *env, const std::string &exceptionClass, F &&f) {
  const SignalGuard guard;
  return wrapException(env, exceptionClass, std::forward<F>(f));
}
