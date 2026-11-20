#pragma once
#include <stdexcept>

#include <jni.h>
namespace polyregion::generated::registry::Natives {

[[maybe_unused]] jlong dynamicLibraryLoad0(JNIEnv *env, jclass, jstring name);
[[maybe_unused]] void dynamicLibraryRelease0(JNIEnv *env, jclass, jlong handle);
[[maybe_unused]] void registerFilesToDropOnUnload0(JNIEnv *env, jclass, jobject file);

static void unregisterMethods(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/Natives");
  if (!found) return env->ExceptionClear();
  const jint result = env->UnregisterNatives(found);
  env->DeleteLocalRef(found);
  if (result != 0) {
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/Natives");
  }
}

static void registerMethods(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/Natives");
  if (!found) {
    throw std::logic_error("Cannot find class polyregion/jvm/Natives");
  }
  const static JNINativeMethod methods[3] = {{.name = const_cast<char *>("dynamicLibraryLoad0"),
                                              .signature = const_cast<char *>("(Ljava/lang/String;)J"),
                                              .fnPtr = reinterpret_cast<void *>(&dynamicLibraryLoad0)},
                                             {.name = const_cast<char *>("dynamicLibraryRelease0"),
                                              .signature = const_cast<char *>("(J)V"),
                                              .fnPtr = reinterpret_cast<void *>(&dynamicLibraryRelease0)},
                                             {.name = const_cast<char *>("registerFilesToDropOnUnload0"),
                                              .signature = const_cast<char *>("(Ljava/io/File;)V"),
                                              .fnPtr = reinterpret_cast<void *>(&registerFilesToDropOnUnload0)}};
  const jint result = env->RegisterNatives(found, methods, 3);
  env->DeleteLocalRef(found);
  if (result != 0) {
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/Natives");
  }
}

} // namespace polyregion::generated::registry::Natives