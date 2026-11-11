#pragma once
#include <stdexcept>

#include <jni.h>
namespace polyregion::generated::registry::Platform {

[[maybe_unused]] jobject createQueue0(JNIEnv *env, jclass, jlong nativePeer, jobject owner, jlong timeoutMillis);
[[maybe_unused]] void deleteDevicePeer0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void deletePlatformPeer0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void deleteQueuePeer0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] jobjectArray deviceFeatures0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] jobjectArray deviceProperties0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] jobjectArray devices0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void enqueueDeviceToHostAsync0(JNIEnv *env, jclass, jlong nativePeer, jlong src, jobject dst, jint size, jobject cb);
[[maybe_unused]] void enqueueHostToDeviceAsync0(JNIEnv *env, jclass, jlong nativePeer, jobject src, jlong dst, jint size, jobject cb);
[[maybe_unused]] void enqueueInvokeAsync0(JNIEnv *env, jclass, jlong nativePeer, jstring moduleName, jstring symbol, jbyteArray argTypes,
                                          jbyteArray argData, jobject policy, jobject cb);
[[maybe_unused]] void free0(JNIEnv *env, jclass, jlong nativePeer, jlong handle);
[[maybe_unused]] void loadModule0(JNIEnv *env, jclass, jlong nativePeer, jstring name, jbyteArray image);
[[maybe_unused]] jlong malloc0(JNIEnv *env, jclass, jlong nativePeer, jlong size, jbyte access);
[[maybe_unused]] jboolean moduleLoaded0(JNIEnv *env, jclass, jlong nativePeer, jstring name);
[[maybe_unused]] jobjectArray runtimeProperties0(JNIEnv *env, jclass, jlong nativePeer);

thread_local jclass clazz = nullptr;

static void unregisterMethods(JNIEnv *env) {
  if (!clazz) return;
  if (env->UnregisterNatives(clazz) != 0) {
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/runtime/Platform");
  }
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) return;
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Platform")));
  const static JNINativeMethod methods[15] = {
      {.name = const_cast<char *>("createQueue0"),
       .signature = const_cast<char *>("(JLpolyregion/jvm/runtime/Device;J)Lpolyregion/jvm/runtime/Device$Queue;"),
       .fnPtr = reinterpret_cast<void *>(&createQueue0)},
      {.name = const_cast<char *>("deleteDevicePeer0"),
       .signature = const_cast<char *>("(J)V"),
       .fnPtr = reinterpret_cast<void *>(&deleteDevicePeer0)},
      {.name = const_cast<char *>("deletePlatformPeer0"),
       .signature = const_cast<char *>("(J)V"),
       .fnPtr = reinterpret_cast<void *>(&deletePlatformPeer0)},
      {.name = const_cast<char *>("deleteQueuePeer0"),
       .signature = const_cast<char *>("(J)V"),
       .fnPtr = reinterpret_cast<void *>(&deleteQueuePeer0)},
      {.name = const_cast<char *>("deviceFeatures0"),
       .signature = const_cast<char *>("(J)[Ljava/lang/String;"),
       .fnPtr = reinterpret_cast<void *>(&deviceFeatures0)},
      {.name = const_cast<char *>("deviceProperties0"),
       .signature = const_cast<char *>("(J)[Lpolyregion/jvm/runtime/Property;"),
       .fnPtr = reinterpret_cast<void *>(&deviceProperties0)},
      {.name = const_cast<char *>("devices0"),
       .signature = const_cast<char *>("(J)[Lpolyregion/jvm/runtime/Device;"),
       .fnPtr = reinterpret_cast<void *>(&devices0)},
      {.name = const_cast<char *>("enqueueDeviceToHostAsync0"),
       .signature = const_cast<char *>("(JJLjava/nio/ByteBuffer;ILjava/lang/Runnable;)V"),
       .fnPtr = reinterpret_cast<void *>(&enqueueDeviceToHostAsync0)},
      {.name = const_cast<char *>("enqueueHostToDeviceAsync0"),
       .signature = const_cast<char *>("(JLjava/nio/ByteBuffer;JILjava/lang/Runnable;)V"),
       .fnPtr = reinterpret_cast<void *>(&enqueueHostToDeviceAsync0)},
      {.name = const_cast<char *>("enqueueInvokeAsync0"),
       .signature = const_cast<char *>("(JLjava/lang/String;Ljava/lang/String;[B[BLpolyregion/jvm/runtime/Policy;Ljava/lang/Runnable;)V"),
       .fnPtr = reinterpret_cast<void *>(&enqueueInvokeAsync0)},
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
  if (env->RegisterNatives(clazz, methods, 15) != 0) {
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/runtime/Platform");
  }
}

} // namespace polyregion::generated::registry::Platform