#pragma once
#include <jni.h>
#include <stdexcept>
namespace polyregion::generated::registry::Natives {

[[maybe_unused]] jlong dynamicLibraryLoad0(JNIEnv *env, jclass, jstring name);
[[maybe_unused]] void dynamicLibraryRelease0(JNIEnv *env, jclass, jlong handle);
[[maybe_unused]] void registerFilesToDropOnUnload0(JNIEnv *env, jclass, jobject file);

thread_local jclass clazz = nullptr;

static void unregisterMethods(JNIEnv *env) {
  if (!clazz) return;
  if(env->UnregisterNatives(clazz) != 0){
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/Natives");
  }
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) return;
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/Natives")));
  const static JNINativeMethod methods[3] = {
      {(char *)"dynamicLibraryLoad0", (char *)"(Ljava/lang/String;)J", (void *)&dynamicLibraryLoad0},
      {(char *)"dynamicLibraryRelease0", (char *)"(J)V", (void *)&dynamicLibraryRelease0},
      {(char *)"registerFilesToDropOnUnload0", (char *)"(Ljava/io/File;)V", (void *)&registerFilesToDropOnUnload0}};
  if(env->RegisterNatives(clazz, methods, 3) != 0){
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/Natives");
  }
}

} // namespace polyregion::generated::registry::Natives