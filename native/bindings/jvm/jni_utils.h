#pragma once
#include "jni.h"

#include "variants.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>

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
  auto ys = env->NewObjectArray(jsize(xs.size()), clazz, nullptr);
  for (jsize i = 0; i < jsize(xs.size()); ++i)
    env->SetObjectArrayElement(ys, i, f(xs[i]));
  return ys;
}

static std::string fromJni(JNIEnv *env, const jstring &s) {
  auto data = env->GetStringUTFChars(s, nullptr);
  std::string out(data);
  env->ReleaseStringUTFChars(s, data);
  return out;
}
template <typename T, typename = typename std::enable_if<std::is_convertible_v<T, char>>>
static std::vector<T> fromJni(JNIEnv *env, jbyteArray xs) {
  auto xsData = env->GetByteArrayElements(xs, nullptr);
  auto bytes = std::vector<T>(xsData, xsData + env->GetArrayLength(xs));
  env->ReleaseByteArrayElements(xs, xsData, JNI_ABORT);
  return bytes;
}

static std::vector<jlong> fromJni(JNIEnv *env, jlongArray xs) {
  std::vector<jlong> ys(env->GetArrayLength(xs));
  env->GetLongArrayRegion(xs, 0, jsize(ys.size()), ys.data());
  return ys;
}

template <typename T = std::nullptr_t>
static T throwGeneric(JNIEnv *env, const std::string &exceptionClass, const std::string &message) {
  if (auto exClass = env->FindClass(exceptionClass.c_str()); exClass) {
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
