#include <string>
#include <vector>

#include "compiler.h"
#include "jni_utils.h"
#include "mirror.h"
#include "polyregion_jvm_compiler_Compiler.h"

static constexpr const char *EX = "polyregion/PolyregionCompilerException";

using namespace polyregion;
namespace gen = ::generated;

[[maybe_unused]] jint JNI_OnLoad(JavaVM *vm, void *) {
  fprintf(stderr, "JVM enter\n");
  JNIEnv *env = getEnv(vm);
  if (!env) return JNI_ERR;
  compiler::initialise();
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

static generated::Layout::Instance toJni(JNIEnv *env, const compiler::Layout &l) {
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

static generated::Event::Instance toJni(JNIEnv *env, const compiler::Event &e) {
  return gen::Event::of(env)(env, e.epochMillis, e.elapsedNanos, toJni(env, e.name), toJni(env, e.data));
}

static compiler::Options fromJni(JNIEnv *env, jobject options) {
  auto opt = gen::Options::of(env).wrap(env, options);
  auto targetOrdinal = opt.target(env);
  if (auto target = polyregion::compiler::targetFromOrdinal(targetOrdinal); target) {
    return {.target = *target, .arch = fromJni(env, opt.arch(env))};
  } else
    throw std::logic_error("Unknown target value: " + std::to_string(targetOrdinal));
}

[[maybe_unused]] jobject Java_polyregion_jvm_compiler_Compiler_layoutOf(JNIEnv *env, jclass, //
                                                                        jbyteArray structDef, jobject options) {
  return wrapException(env, EX, [&]() {
    return toJni(env, compiler::layoutOf(fromJni<char>(env, structDef), fromJni(env, options))).instance;
  });
}

[[maybe_unused]] jobject Java_polyregion_jvm_compiler_Compiler_compile(JNIEnv *env, jclass, //
                                                                       jbyteArray function, jboolean emitDisassembly,
                                                                       jobject options) {
  return wrapException(env, EX, [&]() {
    auto c = compiler::compile(fromJni<char>(env, function), fromJni(env, options));
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
