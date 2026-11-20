#pragma once
#include <jni.h>
namespace polyregion::generated::registry::Compiler {

[[maybe_unused]] jobject compile0(JNIEnv *env, jclass, jbyteArray function, jboolean emitAssembly, jobject options, jbyte opt);
[[maybe_unused]] jbyte hostTarget0(JNIEnv *env, jclass);
[[maybe_unused]] jstring hostTriplet0(JNIEnv *env, jclass);
[[maybe_unused]] jobjectArray layoutOf0(JNIEnv *env, jclass, jbyteArray structDef, jobject options);
[[maybe_unused]] jbyteArray loadPackage0(JNIEnv *env, jclass, jstring name, jstring roots, jstring capabilities);

static void unregisterMethods(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Compiler");
  if (!found) {
    env->ExceptionClear();
    return;
  }
  if (env->UnregisterNatives(found) != 0) env->ExceptionClear();
  env->DeleteLocalRef(found);
}

[[nodiscard]] static bool registerMethods(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Compiler");
  if (!found) return false;
  const static JNINativeMethod methods[5] = {
      {.name = const_cast<char *>("compile0"),
       .signature = const_cast<char *>("([BZLpolyregion/jvm/compiler/Options;B)Lpolyregion/jvm/compiler/Compilation;"),
       .fnPtr = reinterpret_cast<void *>(&compile0)},
      {.name = const_cast<char *>("hostTarget0"), .signature = const_cast<char *>("()B"), .fnPtr = reinterpret_cast<void *>(&hostTarget0)},
      {.name = const_cast<char *>("hostTriplet0"),
       .signature = const_cast<char *>("()Ljava/lang/String;"),
       .fnPtr = reinterpret_cast<void *>(&hostTriplet0)},
      {.name = const_cast<char *>("layoutOf0"),
       .signature = const_cast<char *>("([BLpolyregion/jvm/compiler/Options;)[Lpolyregion/jvm/compiler/Layout;"),
       .fnPtr = reinterpret_cast<void *>(&layoutOf0)},
      {.name = const_cast<char *>("loadPackage0"),
       .signature = const_cast<char *>("(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)[B"),
       .fnPtr = reinterpret_cast<void *>(&loadPackage0)}};
  const jint result = env->RegisterNatives(found, methods, 5);
  env->DeleteLocalRef(found);
  return result == 0;
}

} // namespace polyregion::generated::registry::Compiler