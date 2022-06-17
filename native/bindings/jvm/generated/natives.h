#pragma once
#include <jni.h>
namespace polyregion::generated::registry::Natives {

[[maybe_unused]] void dynamicLibraryRelease0(JNIEnv *env, jclass, jlong handle);
[[maybe_unused]] jlongArray pointerOfDirectBuffers0(JNIEnv *env, jclass, jobjectArray buffers);
[[maybe_unused]] jlong pointerOfDirectBuffer0(JNIEnv *env, jclass, jobject buffer);
[[maybe_unused]] jlong dynamicLibraryLoad0(JNIEnv *env, jclass, jstring name);
[[maybe_unused]] void registerFilesToDropOnUnload0(JNIEnv *env, jclass, jobject file);

static jclass clazz{};

static void unregisterMethods(JNIEnv *env) {
  if (!clazz) return;
  env->UnregisterNatives(clazz);
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) unregisterMethods(env);
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/Natives")));
  static JNINativeMethod methods[] = {
      {(char *)"dynamicLibraryRelease0", (char *)"(J)V", (void *)&dynamicLibraryRelease0},
      {(char *)"pointerOfDirectBuffers0", (char *)"([Ljava/nio/Buffer;)[J", (void *)&pointerOfDirectBuffers0},
      {(char *)"pointerOfDirectBuffer0", (char *)"(Ljava/nio/Buffer;)J", (void *)&pointerOfDirectBuffer0},
      {(char *)"dynamicLibraryLoad0", (char *)"(Ljava/lang/String;)J", (void *)&dynamicLibraryLoad0},
      {(char *)"registerFilesToDropOnUnload0", (char *)"(Ljava/io/File;)V", (void *)&registerFilesToDropOnUnload0}};
  env->RegisterNatives(clazz, methods, 5);
}

} // namespace polyregion::generated::registry::Natives