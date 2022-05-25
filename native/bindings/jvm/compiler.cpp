#include <string>
#include <vector>

#include "compiler.h"
#include "jni_utils.h"
#include "mirror.h"
#include "polyregion_jvm_compiler_Compiler.h"

[[noreturn]] static void throwGeneric(JNIEnv *env, const std::string &message) {
  throwGeneric("polyregion/PolyregionCompilerException", env, message);
}

using namespace polyregion;

static std::unique_ptr<generated::Layout> Layout;
static std::unique_ptr<generated::Compilation> Compilation;
static std::unique_ptr<generated::Event> Event;
static std::unique_ptr<generated::Options> Options;
static std::unique_ptr<generated::Member> Member;
static jclass StringClass;

[[maybe_unused]] jint JNI_OnLoad(JavaVM *vm, void *) {
  compiler::initialise();

  JNIEnv *env = getEnv(vm);
  if (!env) return JNI_ERR;

  Layout = std::make_unique<generated::Layout>(env);
  Compilation = std::make_unique<generated::Compilation>(env);
  Event = std::make_unique<generated::Event>(env);
  Options = std::make_unique<generated::Options>(env);
  Member = std::make_unique<generated::Member>(env);
  StringClass = env->FindClass("java/lang/String");

  return JNI_VERSION_1_1;
}

static generated::Member::Instance toJni(JNIEnv *env, const compiler::Member &m) {
  return (*Member)(env, toJni(env, m.name.symbol), jlong(m.offsetInBytes), jlong(m.sizeInBytes));
}

static generated::Layout::Instance toJni(JNIEnv *env, const compiler::Layout &l) {
  return (*Layout)(env,                                                                                  //
                   toJni(env, l.name.fqn, StringClass, [&](auto &x) { return toJni(env, x); }),          //
                   jlong(l.sizeInBytes),                                                                 //
                   jlong(l.alignment),                                                                   //
                   toJni(env, l.members, Member->clazz, [&](auto &x) { return toJni(env, x).instance; }) //
  );
}

static generated::Event::Instance toJni(JNIEnv *env, const compiler::Event &e) {
  return (*Event)(env, e.epochMillis, e.elapsedNanos, toJni(env, e.name), toJni(env, e.data));
}

static compiler::Options fromJni(JNIEnv *env, jobject options) {
  auto opt = Options->wrap(env, options);
  auto targetOrdinal = opt.target(env);
  if (auto target = polyregion::compiler::targetFromOrdinal(targetOrdinal); target) {
    return {.target = *target, .arch = fromJni(env, opt.arch(env))};
  } else
    throwGeneric(env, "Unknown target value: " + std::to_string(targetOrdinal));
}

[[maybe_unused]] jobject Java_polyregion_jvm_compiler_Compiler_layoutOf(JNIEnv *env, jclass, //
                                                                        jbyteArray structDef, jobject options) {
  return toJni(env, compiler::layoutOf(fromJni<char>(env, structDef), fromJni(env, options))).instance;
}
[[maybe_unused]] jobject Java_polyregion_jvm_compiler_Compiler_compile(JNIEnv *env, jclass, //
                                                                       jbyteArray function, jboolean emitDisassembly,
                                                                       jobject options) {

  auto c = compiler::compile(fromJni<char>(env, function), fromJni(env, options));
  if (!c.binary) throwGeneric(env, c.messages);

  auto bin = env->NewByteArray(jsize(c.binary->size()));
  auto binElems = env->GetByteArrayElements(bin, nullptr);
  std::copy(c.binary->begin(), c.binary->end(), binElems);
  env->ReleaseByteArrayElements(bin, binElems, JNI_COMMIT);

  return (*Compilation)(env,
                        bin,                                                                                   //
                        toJni(env, c.events, Event->clazz, [&](auto &e) { return toJni(env, e).instance; }),   //
                        toJni(env, c.layouts, Layout->clazz, [&](auto &e) { return toJni(env, e).instance; }), //
                        toJni(env, c.messages))                                                                //
      .instance;
}
