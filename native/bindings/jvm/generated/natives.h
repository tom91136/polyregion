#pragma once
#include <jni.h>
#include <stdexcept>
namespace polyregion::generated::registry::Natives {

[[maybe_unused]] void dynamicLibraryRelease0(JNIEnv *env, jclass, jlong handle);
[[maybe_unused]] void registerFilesToDropOnUnload0(JNIEnv *env, jclass, jobject file);
[[maybe_unused]] jlong dynamicLibraryLoad0(JNIEnv *env, jclass, jstring name);
[[maybe_unused]] jlong pointerOfDirectBuffer0(JNIEnv *env, jclass, jobject buffer);
[[maybe_unused]] jlongArray pointerOfDirectBuffers0(JNIEnv *env, jclass, jobjectArray buffers);

static jclass clazz{};

static void unregisterMethods(JNIEnv *env) {
  if (!clazz) return;
  if(env->UnregisterNatives(clazz) != 0){
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/Natives");
  }
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) unregisterMethods(env);
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/Natives")));
  const static JNINativeMethod methods[5] = {
      {(char *)"dynamicLibraryRelease0", (char *)"(J)V", (void *)&dynamicLibraryRelease0},
      {(char *)"registerFilesToDropOnUnload0", (char *)"(Ljava/io/File;)V", (void *)&registerFilesToDropOnUnload0},
      {(char *)"dynamicLibraryLoad0", (char *)"(Ljava/lang/String;)J", (void *)&dynamicLibraryLoad0},
      {(char *)"pointerOfDirectBuffer0", (char *)"(Ljava/nio/Buffer;)J", (void *)&pointerOfDirectBuffer0},
      {(char *)"pointerOfDirectBuffers0", (char *)"([Ljava/nio/Buffer;)[J", (void *)&pointerOfDirectBuffers0}};
  if(env->RegisterNatives(clazz, methods, 5) != 0){
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/Natives");
  }
}

} // namespace polyregion::generated::registry::Natives