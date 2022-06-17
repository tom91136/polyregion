#pragma once
#include <jni.h>
namespace polyregion::generated::registry::Platform {

[[maybe_unused]] jlong malloc0(JNIEnv *env, jclass, jlong nativePeer, jlong size, jbyte access);
[[maybe_unused]] void free0(JNIEnv *env, jclass, jlong nativePeer, jlong handle);
[[maybe_unused]] jobjectArray devices0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] jobject createQueue0(JNIEnv *env, jclass, jlong nativePeer, jobject owner);
[[maybe_unused]] void deletePlatformPeer0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void enqueueHostToDeviceAsync0(JNIEnv *env, jclass, jlong nativePeer, jobject src, jlong dst, jint size, jobject cb);
[[maybe_unused]] void enqueueDeviceToHostAsync0(JNIEnv *env, jclass, jlong nativePeer, jlong src, jobject dst, jint size, jobject cb);
[[maybe_unused]] jobjectArray runtimeProperties0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void loadModule0(JNIEnv *env, jclass, jlong nativePeer, jstring name, jbyteArray image);
[[maybe_unused]] jboolean moduleLoaded0(JNIEnv *env, jclass, jlong nativePeer, jstring name);
[[maybe_unused]] jobjectArray deviceProperties0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void deleteQueuePeer0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void deleteDevicePeer0(JNIEnv *env, jclass, jlong nativePeer);
[[maybe_unused]] void enqueueInvokeAsync0(JNIEnv *env, jclass, jlong nativePeer, jstring moduleName, jstring symbol, jbyteArray argTypes, jbyteArray argData, jobject policy, jobject cb);

static jclass clazz{};

static void unregisterMethods(JNIEnv *env) {
  if (!clazz) return;
  env->UnregisterNatives(clazz);
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) unregisterMethods(env);
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Platform")));
  static JNINativeMethod methods[] = {
      {(char *)"malloc0", (char *)"(JJB)J", (void *)&malloc0},
      {(char *)"free0", (char *)"(JJ)V", (void *)&free0},
      {(char *)"devices0", (char *)"(J)[Lpolyregion/jvm/runtime/Device;", (void *)&devices0},
      {(char *)"createQueue0", (char *)"(JLpolyregion/jvm/runtime/Device;)Lpolyregion/jvm/runtime/Device$Queue;", (void *)&createQueue0},
      {(char *)"deletePlatformPeer0", (char *)"(J)V", (void *)&deletePlatformPeer0},
      {(char *)"enqueueHostToDeviceAsync0", (char *)"(JLjava/nio/ByteBuffer;JILjava/lang/Runnable;)V", (void *)&enqueueHostToDeviceAsync0},
      {(char *)"enqueueDeviceToHostAsync0", (char *)"(JJLjava/nio/ByteBuffer;ILjava/lang/Runnable;)V", (void *)&enqueueDeviceToHostAsync0},
      {(char *)"runtimeProperties0", (char *)"(J)[Lpolyregion/jvm/runtime/Property;", (void *)&runtimeProperties0},
      {(char *)"loadModule0", (char *)"(JLjava/lang/String;[B)V", (void *)&loadModule0},
      {(char *)"moduleLoaded0", (char *)"(JLjava/lang/String;)Z", (void *)&moduleLoaded0},
      {(char *)"deviceProperties0", (char *)"(J)[Lpolyregion/jvm/runtime/Property;", (void *)&deviceProperties0},
      {(char *)"deleteQueuePeer0", (char *)"(J)V", (void *)&deleteQueuePeer0},
      {(char *)"deleteDevicePeer0", (char *)"(J)V", (void *)&deleteDevicePeer0},
      {(char *)"enqueueInvokeAsync0", (char *)"(JLjava/lang/String;Ljava/lang/String;[B[BLpolyregion/jvm/runtime/Policy;Ljava/lang/Runnable;)V", (void *)&enqueueInvokeAsync0}};
  env->RegisterNatives(clazz, methods, 14);
}

} // namespace polyregion::generated::registry::Platform