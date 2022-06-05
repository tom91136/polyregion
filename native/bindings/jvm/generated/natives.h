#pragma once
#include <jni.h>
namespace polyregion::generated::registry::Natives {

[[maybe_unused]] jlong pointerOfDirectBuffer0(JNIEnv *env, jclass, jobject buffer);
[[maybe_unused]] jlongArray pointerOfDirectBuffers0(JNIEnv *env, jclass, jobjectArray buffers);
[[maybe_unused]] void dynamicLibraryRelease0(JNIEnv *env, jclass, jlong handle);
[[maybe_unused]] jlong dynamicLibraryLoad0(JNIEnv *env, jclass, jstring name);
[[maybe_unused]] void registerFilesToDropOnUnload0(JNIEnv *env, jclass, jobject file);

static void registerMethods(JNIEnv *env) {
  auto clazz = env->FindClass("polyregion/jvm/Natives");
  static JNINativeMethod methods[] = {
      {(char *)"pointerOfDirectBuffer0", (char *)"(Ljava/nio/Buffer;)J", (void *)&pointerOfDirectBuffer0},
      {(char *)"pointerOfDirectBuffers0", (char *)"([Ljava/nio/Buffer;)[J", (void *)&pointerOfDirectBuffers0},
      {(char *)"dynamicLibraryRelease0", (char *)"(J)V", (void *)&dynamicLibraryRelease0},
      {(char *)"dynamicLibraryLoad0", (char *)"(Ljava/lang/String;)J", (void *)&dynamicLibraryLoad0},
      {(char *)"registerFilesToDropOnUnload0", (char *)"(Ljava/io/File;)V", (void *)&registerFilesToDropOnUnload0}};
  env->RegisterNatives(clazz, methods, 5);
}

static void unregisterMethods(JNIEnv *env) { env->UnregisterNatives(env->FindClass("polyregion/jvm/Natives")); }
} // namespace polyregion::generated::registry::Natives