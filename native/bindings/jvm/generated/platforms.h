#pragma once
#include <jni.h>
namespace polyregion::generated::registry::Platforms {
static constexpr jbyte TYPE_VOID = 1;
static constexpr jbyte TYPE_BOOL = 2;
static constexpr jbyte TYPE_BYTE = 3;
static constexpr jbyte TYPE_CHAR = 4;
static constexpr jbyte TYPE_SHORT = 5;
static constexpr jbyte TYPE_INT = 6;
static constexpr jbyte TYPE_LONG = 7;
static constexpr jbyte TYPE_FLOAT = 8;
static constexpr jbyte TYPE_DOUBLE = 9;
static constexpr jbyte TYPE_PTR = 10;
static constexpr jbyte ACCESS_RW = 1;
static constexpr jbyte ACCESS_RO = 2;
static constexpr jbyte ACCESS_WO = 3;
[[maybe_unused]] jobject Dynamic0(JNIEnv *env, jclass);
[[maybe_unused]] jobject OpenCL0(JNIEnv *env, jclass);
[[maybe_unused]] jobject HIP0(JNIEnv *env, jclass);
[[maybe_unused]] jobject CUDA0(JNIEnv *env, jclass);
[[maybe_unused]] jobject Relocatable0(JNIEnv *env, jclass);
[[maybe_unused]] void deleteAllPeer0(JNIEnv *env, jclass);

static void registerMethods(JNIEnv *env) {
  auto clazz = env->FindClass("polyregion/jvm/runtime/Platforms");
  static JNINativeMethod methods[] = {
      {(char *)"Dynamic0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&Dynamic0},
      {(char *)"OpenCL0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&OpenCL0},
      {(char *)"HIP0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&HIP0},
      {(char *)"CUDA0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&CUDA0},
      {(char *)"Relocatable0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&Relocatable0},
      {(char *)"deleteAllPeer0", (char *)"()V", (void *)&deleteAllPeer0}};
  env->RegisterNatives(clazz, methods, 6);
}

static void unregisterMethods(JNIEnv *env) { env->UnregisterNatives(env->FindClass("polyregion/jvm/runtime/Platforms")); }
} // namespace polyregion::generated::registry::Platforms