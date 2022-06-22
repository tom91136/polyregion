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
[[maybe_unused]] jobject HIP0(JNIEnv *env, jclass);
[[maybe_unused]] jobject HSA0(JNIEnv *env, jclass);
[[maybe_unused]] jobject OpenCL0(JNIEnv *env, jclass);
[[maybe_unused]] jobject Dynamic0(JNIEnv *env, jclass);
[[maybe_unused]] jobject Relocatable0(JNIEnv *env, jclass);
[[maybe_unused]] void deleteAllPeer0(JNIEnv *env, jclass);

static jclass clazz{};

static void unregisterMethods(JNIEnv *env) {
  fprintf(stderr, "Drop Platforms\n");
  if (!clazz) return;
  if(env->UnregisterNatives(clazz) != 0){
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/runtime/Platforms");
  }
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) unregisterMethods(env);
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Platforms")));
  const static JNINativeMethod methods[7] = {
      {(char *)"CUDA0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&CUDA0},
      {(char *)"HIP0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&HIP0},
      {(char *)"HSA0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&HSA0},
      {(char *)"OpenCL0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&OpenCL0},
      {(char *)"Dynamic0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&Dynamic0},
      {(char *)"Relocatable0", (char *)"()Lpolyregion/jvm/runtime/Platform;", (void *)&Relocatable0},
      {(char *)"deleteAllPeer0", (char *)"()V", (void *)&deleteAllPeer0}};
  if(env->RegisterNatives(clazz, methods, 7) != 0){
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/runtime/Platforms");
  }
}

} // namespace polyregion::generated::registry::Platforms