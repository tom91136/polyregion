#pragma once
#include "jni.h"

#include "variants.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>

static constexpr const char *StringArraySignature = "[Ljava/lang/String;";
static constexpr const char *StringSignature = "Ljava/lang/String;";
static constexpr const char *LongSignature = "J";
static constexpr const char *ByteSignature = "B";
static constexpr const char *ByteArraySignature = "[B";

static JNIEnv *getEnv(JavaVM *vm) {
  JNIEnv *env{};
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_1) != JNI_OK) {
    return nullptr;
  }
  return env;
}

static jstring toJni(JNIEnv *env, const std::string &s) { return env->NewStringUTF(s.c_str()); }


template <typename T, typename F>
static jobjectArray toJni(JNIEnv *env, const std::vector<T> &xs, jclass clazz, const F && f) {
  auto ys = env->NewObjectArray(jsize(xs.size()), clazz, nullptr);
  for (size_t i = 0; i < xs.size(); ++i)
    env->SetObjectArrayElement(ys, i, f(xs[i]));
  return ys;
}

static std::string fromJni(JNIEnv *env, const jstring &s) {
  auto data = env->GetStringUTFChars(s, nullptr);
  std::string out(data);
  env->ReleaseStringUTFChars(s, data);
  return out;
}
 template <typename  T, typename = typename  std::enable_if<std::is_convertible_v<T, char>> >
static std::vector<T> fromJni(JNIEnv *env, jbyteArray xs) {
  auto xsData = env->GetByteArrayElements(xs, nullptr);
  auto bytes = std::vector<T>(xsData, xsData + env->GetArrayLength(xs));
  env->ReleaseByteArrayElements(xs, xsData, JNI_ABORT);
  return bytes;
}

[[noreturn]] static void throwGeneric(const std::string &exceptionClass, JNIEnv *env, const std::string &message) {
  if (auto exClass = env->FindClass(exceptionClass.c_str()); exClass) {
    if (env->ThrowNew(exClass, message.c_str()) != JNI_OK) {
      throw std::logic_error("Cannot throw exception of class " + exceptionClass + ", message was: " + message);
    } else {
      // no return, control transferred back to JVM
    }
  } else {
    throw std::logic_error("Cannot throw exception of unknown class " + exceptionClass + ", message was: " + message);
  }
}
