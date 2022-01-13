#include <string>
#include <vector>
#include <iostream>

#include "compiler.h"
#include "polyregion_PolyregionCompiler.h"
#include "polyregion_compiler.h"

static constexpr const char *StringSignature = "Ljava/lang/String;";

static void throwGeneric(JNIEnv *env, const std::string &message) {
  if (auto exClass = env->FindClass("polyregion/PolyregionCompilerException"); exClass) {
    env->ThrowNew(exClass, message.c_str());
  }
}

static std::pair<jobject, jclass> newNoArgObject(JNIEnv *env, const std::string &name) {
  if (auto clazz = env->FindClass(name.c_str()); clazz) {
    auto ctor = env->GetMethodID(clazz, "<init>", "()V"); // no parameters
    return {env->NewObject(clazz, ctor), clazz};
  } else
    return {nullptr, nullptr};
}

// public final class Elapsed {
//   String name;
//   long nanos;
// }
// public final class Compilation {
//   byte[] program;
//   String disassembly;
//   String messages;
//   Elapsed[] elapsed;
// }

jobject Java_polyregion_PolyregionCompiler_compile(JNIEnv *env, jclass thisCls, jbyteArray ast,
                                                   jboolean emitDisassembly, jshort backend) {


  auto astData = env->GetByteArrayElements(ast, nullptr);
  auto c = polyregion::compiler::compile(std::vector<uint8_t>(astData, astData + env->GetArrayLength(ast)));
  env->ReleaseByteArrayElements(ast, astData, JNI_ABORT);

  std::cout << "Compile complete" <<std::endl;


  auto [compilation, compilationCls] = newNoArgObject(env, "polyregion/Compilation");
  auto programField = env->GetFieldID(compilationCls, "program", "[B");
  auto messagesField = env->GetFieldID(compilationCls, "messages", StringSignature);
  auto disassemblyField = env->GetFieldID(compilationCls, "disassembly", StringSignature);
  std::cout << compilation <<std::endl;
  // FIXME field not found!?
  //  auto elapsedField = env->GetFieldID(compilationCls, "elapsed", "[Lpolyregion/Elapsed");

  if (c.binary) {
    auto bin = env->NewByteArray(jsize(c.binary->size()));
    auto binElems = env->GetByteArrayElements(bin, nullptr);
    std::copy(c.binary->begin(), c.binary->end(), binElems);
    env->ReleaseByteArrayElements(bin, binElems, JNI_COMMIT);
  }

  if (c.disassembly) {
    env->SetObjectField(compilation, disassemblyField, env->NewStringUTF(c.disassembly->c_str()));
  }

  env->SetObjectField(compilation, messagesField, env->NewStringUTF(c.messages.c_str()));

  auto [elapsed, elapsedCls] = newNoArgObject(env, "polyregion/Elapsed");
  auto nameField = env->GetFieldID(elapsedCls, "name", StringSignature);
  auto nanosField = env->GetFieldID(elapsedCls, "nanos", "J");
  auto elapsedElems = env->NewObjectArray(jint(c.elapsed.size()), elapsedCls, elapsed);
  for (jsize i = 0; i < c.elapsed.size(); ++i) {
    auto elem = env->GetObjectArrayElement(elapsedElems, i);
    env->SetObjectField(elem, nameField, env->NewStringUTF(c.elapsed[i].first.c_str()));
    env->SetLongField(elem, nanosField, jlong(c.elapsed[i].second));
  }

  return compilation;
}

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  polyregion::compiler::initialise();
  return JNI_VERSION_1_1;
}