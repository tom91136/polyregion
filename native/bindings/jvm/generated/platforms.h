#pragma once
#include <stdexcept>

#include <jni.h>
namespace polyregion::generated::registry::Platforms {

[[maybe_unused]] jobject CUDA0(JNIEnv *env, jclass);
[[maybe_unused]] jobject Dynamic0(JNIEnv *env, jclass);
[[maybe_unused]] jobject HIP0(JNIEnv *env, jclass);
[[maybe_unused]] jobject HSA0(JNIEnv *env, jclass);
[[maybe_unused]] jobject LevelZero0(JNIEnv *env, jclass);
[[maybe_unused]] jobject Metal0(JNIEnv *env, jclass);
[[maybe_unused]] jobject OpenCL0(JNIEnv *env, jclass);
[[maybe_unused]] jobject Relocatable0(JNIEnv *env, jclass);
[[maybe_unused]] jobject Vulkan0(JNIEnv *env, jclass);
[[maybe_unused]] void deleteAllPeers0(JNIEnv *env, jclass);
[[maybe_unused]] jobject directBufferFromPointer0(JNIEnv *env, jclass, jlong ptr, jlong size);
[[maybe_unused]] jlong pointerOfDirectBuffer0(JNIEnv *env, jclass, jobject buffer);
[[maybe_unused]] jlongArray pointerOfDirectBuffers0(JNIEnv *env, jclass, jobjectArray buffers);

static void unregisterMethods(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Platforms");
  if (!found) return env->ExceptionClear();
  const jint result = env->UnregisterNatives(found);
  env->DeleteLocalRef(found);
  if (result != 0) {
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/runtime/Platforms");
  }
}

static void registerMethods(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Platforms");
  if (!found) {
    throw std::logic_error("Cannot find class polyregion/jvm/runtime/Platforms");
  }
  const static JNINativeMethod methods[13] = {{.name = const_cast<char *>("CUDA0"),
                                               .signature = const_cast<char *>("()Lpolyregion/jvm/runtime/Platform;"),
                                               .fnPtr = reinterpret_cast<void *>(&CUDA0)},
                                              {.name = const_cast<char *>("Dynamic0"),
                                               .signature = const_cast<char *>("()Lpolyregion/jvm/runtime/Platform;"),
                                               .fnPtr = reinterpret_cast<void *>(&Dynamic0)},
                                              {.name = const_cast<char *>("HIP0"),
                                               .signature = const_cast<char *>("()Lpolyregion/jvm/runtime/Platform;"),
                                               .fnPtr = reinterpret_cast<void *>(&HIP0)},
                                              {.name = const_cast<char *>("HSA0"),
                                               .signature = const_cast<char *>("()Lpolyregion/jvm/runtime/Platform;"),
                                               .fnPtr = reinterpret_cast<void *>(&HSA0)},
                                              {.name = const_cast<char *>("LevelZero0"),
                                               .signature = const_cast<char *>("()Lpolyregion/jvm/runtime/Platform;"),
                                               .fnPtr = reinterpret_cast<void *>(&LevelZero0)},
                                              {.name = const_cast<char *>("Metal0"),
                                               .signature = const_cast<char *>("()Lpolyregion/jvm/runtime/Platform;"),
                                               .fnPtr = reinterpret_cast<void *>(&Metal0)},
                                              {.name = const_cast<char *>("OpenCL0"),
                                               .signature = const_cast<char *>("()Lpolyregion/jvm/runtime/Platform;"),
                                               .fnPtr = reinterpret_cast<void *>(&OpenCL0)},
                                              {.name = const_cast<char *>("Relocatable0"),
                                               .signature = const_cast<char *>("()Lpolyregion/jvm/runtime/Platform;"),
                                               .fnPtr = reinterpret_cast<void *>(&Relocatable0)},
                                              {.name = const_cast<char *>("Vulkan0"),
                                               .signature = const_cast<char *>("()Lpolyregion/jvm/runtime/Platform;"),
                                               .fnPtr = reinterpret_cast<void *>(&Vulkan0)},
                                              {.name = const_cast<char *>("deleteAllPeers0"),
                                               .signature = const_cast<char *>("()V"),
                                               .fnPtr = reinterpret_cast<void *>(&deleteAllPeers0)},
                                              {.name = const_cast<char *>("directBufferFromPointer0"),
                                               .signature = const_cast<char *>("(JJ)Ljava/nio/ByteBuffer;"),
                                               .fnPtr = reinterpret_cast<void *>(&directBufferFromPointer0)},
                                              {.name = const_cast<char *>("pointerOfDirectBuffer0"),
                                               .signature = const_cast<char *>("(Ljava/nio/Buffer;)J"),
                                               .fnPtr = reinterpret_cast<void *>(&pointerOfDirectBuffer0)},
                                              {.name = const_cast<char *>("pointerOfDirectBuffers0"),
                                               .signature = const_cast<char *>("([Ljava/nio/Buffer;)[J"),
                                               .fnPtr = reinterpret_cast<void *>(&pointerOfDirectBuffers0)}};
  const jint result = env->RegisterNatives(found, methods, 13);
  env->DeleteLocalRef(found);
  if (result != 0) {
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/runtime/Platforms");
  }
}

} // namespace polyregion::generated::registry::Platforms