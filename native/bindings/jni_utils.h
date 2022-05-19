#pragma once
#include "jni.h"

#include <stdexcept>
#include <string>

static constexpr const char *StringArraySignature = "[Ljava/lang/String;";
static constexpr const char *StringSignature = "Ljava/lang/String;";
static constexpr const char *LongSignature = "J";
static constexpr const char *ByteSignature = "B";
static constexpr const char *ByteArraySignature = "[B";

static std::string copyString(JNIEnv *env, jstring str) {
  auto data = env->GetStringUTFChars(str, nullptr);
  std::string out(data);
  env->ReleaseStringUTFChars(str, data);
  return out;
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

static jobject newNoArgObject(JNIEnv *env, jclass clazz) {
  auto ctor = env->GetMethodID(clazz, "<init>", "()V"); // no parameters
  return env->NewObject(clazz, ctor);
}
