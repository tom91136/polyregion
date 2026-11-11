#pragma once
#include <stdexcept>

#include <jni.h>
namespace polyregion::generated::registry::Compiler {

[[maybe_unused]] jobject compile0(JNIEnv *env, jclass, jbyteArray function, jboolean emitAssembly, jobject options, jbyte opt);
[[maybe_unused]] jbyte hostTarget0(JNIEnv *env, jclass);
[[maybe_unused]] jstring hostTriplet0(JNIEnv *env, jclass);
[[maybe_unused]] jobjectArray layoutOf0(JNIEnv *env, jclass, jbyteArray structDef, jobject options);

thread_local jclass clazz = nullptr;

static void unregisterMethods(JNIEnv *env) {
  if (!clazz) return;
  if (env->UnregisterNatives(clazz) != 0) {
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/compiler/Compiler");
  }
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) return;
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/compiler/Compiler")));
  const static JNINativeMethod methods[4] = {
      {.name = const_cast<char *>("compile0"),
       .signature = const_cast<char *>("([BZLpolyregion/jvm/compiler/Options;B)Lpolyregion/jvm/compiler/Compilation;"),
       .fnPtr = reinterpret_cast<void *>(&compile0)},
      {.name = const_cast<char *>("hostTarget0"), .signature = const_cast<char *>("()B"), .fnPtr = reinterpret_cast<void *>(&hostTarget0)},
      {.name = const_cast<char *>("hostTriplet0"),
       .signature = const_cast<char *>("()Ljava/lang/String;"),
       .fnPtr = reinterpret_cast<void *>(&hostTriplet0)},
      {.name = const_cast<char *>("layoutOf0"),
       .signature = const_cast<char *>("([BLpolyregion/jvm/compiler/Options;)[Lpolyregion/jvm/compiler/Layout;"),
       .fnPtr = reinterpret_cast<void *>(&layoutOf0)}};
  if (env->RegisterNatives(clazz, methods, 4) != 0) {
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/compiler/Compiler");
  }
}

} // namespace polyregion::generated::registry::Compiler