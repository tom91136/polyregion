#pragma once
#include <jni.h>
#include <stdexcept>
namespace polyregion::generated::registry::Platforms {
static constexpr jbyte ACCESS_RO = 2;
static constexpr jbyte ACCESS_RW = 1;
static constexpr jbyte ACCESS_WO = 3;
static constexpr jbyte TYPE_BOOL = 2;
static constexpr jbyte TYPE_BYTE = 3;
static constexpr jbyte TYPE_CHAR = 4;
static constexpr jbyte TYPE_DOUBLE = 9;
static constexpr jbyte TYPE_FLOAT = 8;
static constexpr jbyte TYPE_INT = 6;
static constexpr jbyte TYPE_LONG = 7;
static constexpr jbyte TYPE_PTR = 10;
static constexpr jbyte TYPE_SHORT = 5;
static constexpr jbyte TYPE_VOID = 1;
[[maybe_unused]] jobject CUDA0(JNIEnv *env, jclass);
[[maybe_unused]] jobject Dynamic0(JNIEnv *env, jclass);
[[maybe_unused]] jobject HIP0(JNIEnv *env, jclass);
[[maybe_unused]] jobject HSA0(JNIEnv *env, jclass);
[[maybe_unused]] jobject OpenCL0(JNIEnv *env, jclass);
[[maybe_unused]] jobject Relocatable0(JNIEnv *env, jclass);
[[maybe_unused]] void deleteAllPeers0(JNIEnv *env, jclass);
[[maybe_unused]] jlong pointerOfDirectBuffer0(JNIEnv *env, jclass, jobject buffer);
[[maybe_unused]] jlongArray pointerOfDirectBuffers0(JNIEnv *env, jclass, jobjectArray buffers);

thread_local jclass clazz = nullptr;

static void unregisterMethods(JNIEnv *env) {
  if (!clazz) return;
  if(env->UnregisterNatives(clazz) != 0){
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/runtime/Platforms");
  }
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) return;
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Platforms")));
  const static JNINativeMethod methods[9] = {
      {(char *)"CUDA0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&CUDA0},
      {(char *)"Dynamic0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&Dynamic0},
      {(char *)"HIP0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&HIP0},
      {(char *)"HSA0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&HSA0},
      {(char *)"OpenCL0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&OpenCL0},
      {(char *)"Relocatable0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&Relocatable0},
      {(char *)"deleteAllPeers0", (char *)"()V", (void *)&deleteAllPeers0},
      {(char *)"pointerOfDirectBuffer0", (char *)"(Ljava/nio/Buffer;)J", (void *)&pointerOfDirectBuffer0},
      {(char *)"pointerOfDirectBuffers0", (char *)"([Ljava/nio/Buffer;)[J", (void *)&pointerOfDirectBuffers0}};
  if(env->RegisterNatives(clazz, methods, 9) != 0){
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/runtime/Platforms");
  }
}

} // namespace polyregion::generated::registry::Platforms