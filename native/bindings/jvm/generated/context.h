#pragma once
#include <stdexcept>

#include <jni.h>
namespace polyregion::generated::registry::Context {

[[maybe_unused]] jlong address0(JNIEnv *env, jclass, jobject buffer);
[[maybe_unused]] void callHost0(JNIEnv *env, jclass, jlong context, jobjectArray imageNames, jbyteArray imageTargets,
                                jobjectArray imageFeatures, jobjectArray images, jstring programName, jstring entry, jbyteArray argTypes,
                                jbyteArray argData);
[[maybe_unused]] void copyFromDevice0(JNIEnv *env, jclass, jlong context, jlong src, jobject dst, jlong bytes);
[[maybe_unused]] void copyFromDeviceAsync0(JNIEnv *env, jclass, jlong context, jlong src, jobject dst, jlong bytes, jobject completion);
[[maybe_unused]] void copyToDevice0(JNIEnv *env, jclass, jlong context, jobject src, jlong dst, jlong bytes);
[[maybe_unused]] void copyToDeviceAsync0(JNIEnv *env, jclass, jlong context, jobject src, jlong dst, jlong bytes, jobject completion);
[[maybe_unused]] jlong create0(JNIEnv *env, jclass, jint platform, jint device);
[[maybe_unused]] jlong current0(JNIEnv *env, jclass);
[[maybe_unused]] void destroy0(JNIEnv *env, jclass, jlong context);
[[maybe_unused]] jlongArray deviceInfo0(JNIEnv *env, jclass, jint platform, jint device);
[[maybe_unused]] jstring deviceName0(JNIEnv *env, jclass, jint platform, jint device);
[[maybe_unused]] void free0(JNIEnv *env, jclass, jlong context, jlong pointer);
[[maybe_unused]] jboolean hostProgramLoaded0(JNIEnv *env, jclass, jstring name);
[[maybe_unused]] void launch0(JNIEnv *env, jclass, jlong context, jobjectArray imageNames, jbyteArray imageTargets,
                              jobjectArray imageFeatures, jobjectArray images, jstring moduleName, jstring kernel, jbyteArray argTypes,
                              jbyteArray argData, jlongArray dims);
[[maybe_unused]] jboolean libjsigLoaded0(JNIEnv *env, jclass);
[[maybe_unused]] void loadHostProgram0(JNIEnv *env, jclass, jstring name, jbyteArray image);
[[maybe_unused]] jlong malloc0(JNIEnv *env, jclass, jlong context, jlong bytes);
[[maybe_unused]] jint platformCount0(JNIEnv *env, jclass);
[[maybe_unused]] jlongArray platformInfo0(JNIEnv *env, jclass, jint platform);
[[maybe_unused]] jstring platformName0(JNIEnv *env, jclass, jint platform);
[[maybe_unused]] jint select0(JNIEnv *env, jclass, jlong context, jobjectArray imageNames, jbyteArray imageTargets,
                              jobjectArray imageFeatures, jobjectArray images);
[[maybe_unused]] jboolean sharedAddressSpace0(JNIEnv *env, jclass, jlong context);
[[maybe_unused]] jlongArray stats0(JNIEnv *env, jclass, jlong context);

thread_local jclass clazz = nullptr;

static void unregisterMethods(JNIEnv *env) {
  if (!clazz) return;
  if (env->UnregisterNatives(clazz) != 0) {
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/runtime/Context");
  }
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) return;
  auto *found = env->FindClass("polyregion/jvm/runtime/Context");
  if (!found) {
    throw std::logic_error("Cannot find class polyregion/jvm/runtime/Context");
  }
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(found));
  const static JNINativeMethod methods[23] = {
      {.name = const_cast<char *>("address0"),
       .signature = const_cast<char *>("(Ljava/nio/ByteBuffer;)J"),
       .fnPtr = reinterpret_cast<void *>(&address0)},
      {.name = const_cast<char *>("callHost0"),
       .signature = const_cast<char *>("(J[Ljava/lang/String;[B[[Ljava/lang/String;[[BLjava/lang/String;Ljava/lang/String;[B[B)V"),
       .fnPtr = reinterpret_cast<void *>(&callHost0)},
      {.name = const_cast<char *>("copyFromDevice0"),
       .signature = const_cast<char *>("(JJLjava/nio/ByteBuffer;J)V"),
       .fnPtr = reinterpret_cast<void *>(&copyFromDevice0)},
      {.name = const_cast<char *>("copyFromDeviceAsync0"),
       .signature = const_cast<char *>("(JJLjava/nio/ByteBuffer;JLjava/lang/Runnable;)V"),
       .fnPtr = reinterpret_cast<void *>(&copyFromDeviceAsync0)},
      {.name = const_cast<char *>("copyToDevice0"),
       .signature = const_cast<char *>("(JLjava/nio/ByteBuffer;JJ)V"),
       .fnPtr = reinterpret_cast<void *>(&copyToDevice0)},
      {.name = const_cast<char *>("copyToDeviceAsync0"),
       .signature = const_cast<char *>("(JLjava/nio/ByteBuffer;JJLjava/lang/Runnable;)V"),
       .fnPtr = reinterpret_cast<void *>(&copyToDeviceAsync0)},
      {.name = const_cast<char *>("create0"), .signature = const_cast<char *>("(II)J"), .fnPtr = reinterpret_cast<void *>(&create0)},
      {.name = const_cast<char *>("current0"), .signature = const_cast<char *>("()J"), .fnPtr = reinterpret_cast<void *>(&current0)},
      {.name = const_cast<char *>("destroy0"), .signature = const_cast<char *>("(J)V"), .fnPtr = reinterpret_cast<void *>(&destroy0)},
      {.name = const_cast<char *>("deviceInfo0"),
       .signature = const_cast<char *>("(II)[J"),
       .fnPtr = reinterpret_cast<void *>(&deviceInfo0)},
      {.name = const_cast<char *>("deviceName0"),
       .signature = const_cast<char *>("(II)Ljava/lang/String;"),
       .fnPtr = reinterpret_cast<void *>(&deviceName0)},
      {.name = const_cast<char *>("free0"), .signature = const_cast<char *>("(JJ)V"), .fnPtr = reinterpret_cast<void *>(&free0)},
      {.name = const_cast<char *>("hostProgramLoaded0"),
       .signature = const_cast<char *>("(Ljava/lang/String;)Z"),
       .fnPtr = reinterpret_cast<void *>(&hostProgramLoaded0)},
      {.name = const_cast<char *>("launch0"),
       .signature = const_cast<char *>("(J[Ljava/lang/String;[B[[Ljava/lang/String;[[BLjava/lang/String;Ljava/lang/String;[B[B[J)V"),
       .fnPtr = reinterpret_cast<void *>(&launch0)},
      {.name = const_cast<char *>("libjsigLoaded0"),
       .signature = const_cast<char *>("()Z"),
       .fnPtr = reinterpret_cast<void *>(&libjsigLoaded0)},
      {.name = const_cast<char *>("loadHostProgram0"),
       .signature = const_cast<char *>("(Ljava/lang/String;[B)V"),
       .fnPtr = reinterpret_cast<void *>(&loadHostProgram0)},
      {.name = const_cast<char *>("malloc0"), .signature = const_cast<char *>("(JJ)J"), .fnPtr = reinterpret_cast<void *>(&malloc0)},
      {.name = const_cast<char *>("platformCount0"),
       .signature = const_cast<char *>("()I"),
       .fnPtr = reinterpret_cast<void *>(&platformCount0)},
      {.name = const_cast<char *>("platformInfo0"),
       .signature = const_cast<char *>("(I)[J"),
       .fnPtr = reinterpret_cast<void *>(&platformInfo0)},
      {.name = const_cast<char *>("platformName0"),
       .signature = const_cast<char *>("(I)Ljava/lang/String;"),
       .fnPtr = reinterpret_cast<void *>(&platformName0)},
      {.name = const_cast<char *>("select0"),
       .signature = const_cast<char *>("(J[Ljava/lang/String;[B[[Ljava/lang/String;[[B)I"),
       .fnPtr = reinterpret_cast<void *>(&select0)},
      {.name = const_cast<char *>("sharedAddressSpace0"),
       .signature = const_cast<char *>("(J)Z"),
       .fnPtr = reinterpret_cast<void *>(&sharedAddressSpace0)},
      {.name = const_cast<char *>("stats0"), .signature = const_cast<char *>("(J)[J"), .fnPtr = reinterpret_cast<void *>(&stats0)}};
  if (env->RegisterNatives(clazz, methods, 23) != 0) {
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/runtime/Context");
  }
}

} // namespace polyregion::generated::registry::Context