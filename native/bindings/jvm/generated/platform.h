#pragma once
#include <stdexcept>

#include <jni.h>
namespace polyregion::generated::registry::Platform {

[[maybe_unused]] void deleteDevicePeer0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void deletePlatformPeer0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] jobjectArray deviceFeatures0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] jobjectArray deviceProperties0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] jobjectArray devices0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void free0(JNIEnv *env, jclass, jlong nativePeer, jlong handle);
[[maybe_unused]] void loadModule0(JNIEnv *env, jclass, jlong nativePeer, jstring name, jbyteArray image);
[[maybe_unused]] jlong malloc0(JNIEnv *env, jclass, jlong nativePeer, jlong size, jbyte access);
[[maybe_unused]] jboolean moduleLoaded0(JNIEnv *env, jclass, jlong nativePeer, jstring name);
[[maybe_unused]] jobjectArray runtimeProperties0(JNIEnv *env, jclass, jlong nativePeer);

static void unregisterMethods(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Platform");
  if (!found) return env->ExceptionClear();
  const jint result = env->UnregisterNatives(found);
  env->DeleteLocalRef(found);
  if (result != 0) {
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/runtime/Platform");
  }
}

static void registerMethods(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Platform");
  if (!found) {
    throw std::logic_error("Cannot find class polyregion/jvm/runtime/Platform");
  }
  const static JNINativeMethod methods[10] = {
      {.name = const_cast<char *>("deleteDevicePeer0"),
       .signature = const_cast<char *>("(J)V"),
       .fnPtr = reinterpret_cast<void *>(&deleteDevicePeer0)},
      {.name = const_cast<char *>("deletePlatformPeer0"),
       .signature = const_cast<char *>("(J)V"),
       .fnPtr = reinterpret_cast<void *>(&deletePlatformPeer0)},
      {.name = const_cast<char *>("deviceFeatures0"),
       .signature = const_cast<char *>("(J)[Ljava/lang/String;"),
       .fnPtr = reinterpret_cast<void *>(&deviceFeatures0)},
      {.name = const_cast<char *>("deviceProperties0"),
       .signature = const_cast<char *>("(J)[Lpolyregion/jvm/runtime/Property;"),
       .fnPtr = reinterpret_cast<void *>(&deviceProperties0)},
      {.name = const_cast<char *>("devices0"),
       .signature = const_cast<char *>("(J)[Lpolyregion/jvm/runtime/Device;"),
       .fnPtr = reinterpret_cast<void *>(&devices0)},
      {.name = const_cast<char *>("free0"), .signature = const_cast<char *>("(JJ)V"), .fnPtr = reinterpret_cast<void *>(&free0)},
      {.name = const_cast<char *>("loadModule0"),
       .signature = const_cast<char *>("(JLjava/lang/String;[B)V"),
       .fnPtr = reinterpret_cast<void *>(&loadModule0)},
      {.name = const_cast<char *>("malloc0"), .signature = const_cast<char *>("(JJB)J"), .fnPtr = reinterpret_cast<void *>(&malloc0)},
      {.name = const_cast<char *>("moduleLoaded0"),
       .signature = const_cast<char *>("(JLjava/lang/String;)Z"),
       .fnPtr = reinterpret_cast<void *>(&moduleLoaded0)},
      {.name = const_cast<char *>("runtimeProperties0"),
       .signature = const_cast<char *>("(J)[Lpolyregion/jvm/runtime/Property;"),
       .fnPtr = reinterpret_cast<void *>(&runtimeProperties0)}};
  const jint result = env->RegisterNatives(found, methods, 10);
  env->DeleteLocalRef(found);
  if (result != 0) {
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/runtime/Platform");
  }
}

} // namespace polyregion::generated::registry::Platform