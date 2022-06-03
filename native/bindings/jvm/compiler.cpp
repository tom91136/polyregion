#include <string>
#include <vector>

#include "backend/llvmc.h"
#include "compiler.h"
#include "jni_utils.h"
#include "mirror.h"
#include "polyregion_jvm_compiler_Compiler.h"
#include "utils.hpp"

static constexpr const char *EX = "polyregion/jvm/compiler/PolyregionCompilerException";

using namespace polyregion;
namespace cp = ::compiler;
namespace gen = ::generated;

static_assert(polyregion::to_underlying(cp::Target::Object_LLVM_HOST) ==
              polyregion_jvm_compiler_Compiler_Target_1Object_1LLVM_1HOST);
static_assert(polyregion::to_underlying(cp::Target::Object_LLVM_x86_64) ==
              polyregion_jvm_compiler_Compiler_Target_1Object_1LLVM_1x86_164);
static_assert(polyregion::to_underlying(cp::Target::Object_LLVM_AArch64) ==
              polyregion_jvm_compiler_Compiler_Target_1Object_1LLVM_1AArch64);
static_assert(polyregion::to_underlying(cp::Target::Object_LLVM_ARM) ==
              polyregion_jvm_compiler_Compiler_Target_1Object_1LLVM_1ARM);

static_assert(polyregion::to_underlying(cp::Target::Object_LLVM_NVPTX64) ==
              polyregion_jvm_compiler_Compiler_Target_1Object_1LLVM_1NVPTX64);
static_assert(polyregion::to_underlying(cp::Target::Object_LLVM_AMDGCN) ==
              polyregion_jvm_compiler_Compiler_Target_1Object_1LLVM_1AMDGCN);
static_assert(polyregion::to_underlying(cp::Target::Object_LLVM_SPIRV64) ==
              polyregion_jvm_compiler_Compiler_Target_1Object_1LLVM_1SPIRV64);

static_assert(polyregion::to_underlying(cp::Target::Source_C_OpenCL1_1) ==
              polyregion_jvm_compiler_Compiler_Target_1Source_1C_1OpenCL1_11);
static_assert(polyregion::to_underlying(cp::Target::Source_C_C11) ==
              polyregion_jvm_compiler_Compiler_Target_1Source_1C_1C11);

[[maybe_unused]] jint JNI_OnLoad(JavaVM *vm, void *) {
  fprintf(stderr, "JVM enter\n");
  JNIEnv *env = getEnv(vm);
  if (!env) return JNI_ERR;
  cp::initialise();
  return JNI_VERSION_1_1;
}

[[maybe_unused]] void JNI_OnUnload(JavaVM *vm, void *) {
  fprintf(stderr, "JVM exit\n");
  JNIEnv *env = getEnv(vm);
  gen::Layout::drop(env);
  gen::Compilation::drop(env);
  gen::Event::drop(env);
  gen::Options::drop(env);
  gen::Member::drop(env);
  gen::String::drop(env);
}

static generated::Layout::Instance toJni(JNIEnv *env, const cp::Layout &l) {
  return gen::Layout::of(env)(
      env,                                                                                        //
      toJni(env, l.name.fqn, gen::String::of(env).clazz, [&](auto &x) { return toJni(env, x); }), //
      jlong(l.sizeInBytes),                                                                       //
      jlong(l.alignment),                                                                         //
      toJni(env, l.members, gen::Member::of(env).clazz,                                           //
            [&](auto &m) {                                                                        //
              return gen::Member::of(env)(env,                                                    //
                                          toJni(env, m.name.symbol),                              //
                                          jlong(m.offsetInBytes),                                 //
                                          jlong(m.sizeInBytes))                                   //
                  .instance;                                                                      //
            })                                                                                    //
  );                                                                                              //
}

static generated::Event::Instance toJni(JNIEnv *env, const cp::Event &e) {
  return gen::Event::of(env)(env, e.epochMillis, e.elapsedNanos, toJni(env, e.name), toJni(env, e.data));
}

static cp::Options fromJni(JNIEnv *env, jobject options) {
  auto opt = gen::Options::of(env).wrap(env, options);
  auto targetOrdinal = opt.target(env);
  if (auto target = cp::targetFromOrdinal(targetOrdinal); target) {
    return {.target = *target, .arch = fromJni(env, opt.arch(env))};
  } else
    throw std::logic_error("Unknown target value: " + std::to_string(targetOrdinal));
}

static cp::Opt fromJni(JNIEnv *env, jbyte optOrdinal) {
  if (auto opt = cp::optFromOrdinal(optOrdinal); opt) return *opt;
  else
    throw std::logic_error("Unknown opt value: " + std::to_string(optOrdinal));
}

[[maybe_unused]] jbyte Java_polyregion_jvm_compiler_Compiler_hostTarget0(JNIEnv *env, jclass) {
  switch (polyregion::backend::llvmc::defaultHostTriple().getArch()) {
    case llvm::Triple::x86_64: return polyregion::to_underlying(cp::Target::Object_LLVM_x86_64);
    case llvm::Triple::aarch64: return polyregion::to_underlying(cp::Target::Object_LLVM_AArch64);
    case llvm::Triple::arm: return polyregion::to_underlying(cp::Target::Object_LLVM_ARM);
    default: return polyregion_jvm_compiler_Compiler_Target_1UNSUPPORTED;
  }
}

[[maybe_unused]] jstring Java_polyregion_jvm_compiler_Compiler_hostTriplet(JNIEnv *env, jclass) {
  return toJni(env, polyregion::backend::llvmc::defaultHostTriple().str());
}

[[maybe_unused]] jobject Java_polyregion_jvm_compiler_Compiler_layoutOf(JNIEnv *env, jclass, //
                                                                        jbyteArray structDef, jobject options) {
  return wrapException(env, EX, [&]() {
    return toJni(env, cp::layoutOf(fromJni<char>(env, structDef), fromJni(env, options))).instance;
  });
}

[[maybe_unused]] jobject Java_polyregion_jvm_compiler_Compiler_compile(JNIEnv *env, jclass, //
                                                                       jbyteArray function, jboolean emitDisassembly,
                                                                       jobject options, jbyte opt) {
  return wrapException(env, EX, [&]() {
    auto c = cp::compile(fromJni<char>(env, function), fromJni(env, options), fromJni(env, opt));
    if (!c.binary) throw std::logic_error(c.messages);
    auto bin = env->NewByteArray(jsize(c.binary->size()));
    env->SetByteArrayRegion(bin, 0, jsize(c.binary->size()), reinterpret_cast<jbyte *>(c.binary->data()));
    return gen::Compilation::of(env)(
               env,
               bin,                                                                                                //
               toJni(env, c.events, gen::Event::of(env).clazz, [&](auto &e) { return toJni(env, e).instance; }),   //
               toJni(env, c.layouts, gen::Layout::of(env).clazz, [&](auto &e) { return toJni(env, e).instance; }), //
               toJni(env, c.messages))                                                                             //
        .instance;
  });
}
