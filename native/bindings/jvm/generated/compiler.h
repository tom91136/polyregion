#pragma once
#include <jni.h>
#include <stdexcept>
namespace polyregion::generated::registry::Compiler {
static constexpr jbyte Opt_O0 = 10;
static constexpr jbyte Opt_O1 = 11;
static constexpr jbyte Opt_O2 = 12;
static constexpr jbyte Opt_O3 = 13;
static constexpr jbyte Opt_Ofast = 14;
static constexpr jbyte Target_Object_LLVM_AArch64 = 12;
static constexpr jbyte Target_Object_LLVM_AMDGCN = 21;
static constexpr jbyte Target_Object_LLVM_ARM = 13;
static constexpr jbyte Target_Object_LLVM_HOST = 10;
static constexpr jbyte Target_Object_LLVM_NVPTX64 = 20;
static constexpr jbyte Target_Object_LLVM_SPIRV64 = 22;
static constexpr jbyte Target_Object_LLVM_x86_64 = 11;
static constexpr jbyte Target_Source_C_C11 = 30;
static constexpr jbyte Target_Source_C_OpenCL1_1 = 31;
static constexpr jbyte Target_UNSUPPORTED = 1;
[[maybe_unused]] jbyte hostTarget0(JNIEnv *env, jclass);
[[maybe_unused]] jstring hostTriplet0(JNIEnv *env, jclass);
[[maybe_unused]] jobject layoutOf0(JNIEnv *env, jclass, jbyteArray structDef, jobject options);
[[maybe_unused]] jobject compile0(JNIEnv *env, jclass, jbyteArray function, jboolean emitAssembly, jobject options, jbyte opt);

static jclass clazz{};

static void unregisterMethods(JNIEnv *env) {
  if (!clazz) return;
  if(env->UnregisterNatives(clazz) != 0){
    throw std::logic_error("UnregisterNatives returned non-zero for polyregion/jvm/compiler/Compiler");
  }
  env->DeleteGlobalRef(clazz);
  clazz = nullptr;
}

static void registerMethods(JNIEnv *env) {
  if (clazz) unregisterMethods(env);
  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/compiler/Compiler")));
  const static JNINativeMethod methods[4] = {
      {(char *)"hostTarget0", (char *)"()B", (void *)&hostTarget0},
      {(char *)"hostTriplet0", (char *)"()Ljava/lang/String;", (void *)&hostTriplet0},
      {(char *)"layoutOf0", (char *)"([BLpolyregion/jvm/compiler/Options;)Lpolyregion/jvm/compiler/Layout;", (void *)&layoutOf0},
      {(char *)"compile0", (char *)"([BZLpolyregion/jvm/compiler/Options;B)Lpolyregion/jvm/compiler/Compilation;", (void *)&compile0}};
  if(env->RegisterNatives(clazz, methods, 4) != 0){
    throw std::logic_error("RegisterNatives returned non-zero for polyregion/jvm/compiler/Compiler");
  }
}

} // namespace polyregion::generated::registry::Compiler