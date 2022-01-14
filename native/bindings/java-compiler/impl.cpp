#include <iostream>
#include <string>
#include <vector>

#include "compiler.h"
#include "polyregion_PolyregionCompiler.h"
#include "polyregion_compiler.h"

static constexpr const char *StringSignature = "Ljava/lang/String;";

static void throwGeneric(JNIEnv *env, const std::string &message) {
  if (auto exClass = env->FindClass("polyregion/PolyregionCompilerException"); exClass) {
    env->ThrowNew(exClass, message.c_str());
  }
}

static jobject newNoArgObject(JNIEnv *env, jclass clazz) {
  auto ctor = env->GetMethodID(clazz, "<init>", "()V"); // no parameters
  return env->NewObject(clazz, ctor);
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

  std::cout << "Compile complete" << std::endl;

  auto compilationCls = env->FindClass("polyregion/Compilation");
  auto compilation = newNoArgObject(env, compilationCls);
  auto programField = env->GetFieldID(compilationCls, "program", "[B");
  auto messagesField = env->GetFieldID(compilationCls, "messages", StringSignature);
  auto disassemblyField = env->GetFieldID(compilationCls, "disassembly", StringSignature);
  auto eventsField = env->GetFieldID(compilationCls, "events", "[Lpolyregion/Event;");
  std::cout << c << std::endl;

  if (c.binary) {
    auto bin = env->NewByteArray(jsize(c.binary->size()));
    auto binElems = env->GetByteArrayElements(bin, nullptr);
    std::copy(c.binary->begin(), c.binary->end(), binElems);
    env->ReleaseByteArrayElements(bin, binElems, JNI_COMMIT);
    env->SetObjectField(compilation, programField, bin);
  }

  if (c.disassembly) {
    env->SetObjectField(compilation, disassemblyField, env->NewStringUTF(c.disassembly->c_str()));
  }

  env->SetObjectField(compilation, messagesField, env->NewStringUTF(c.messages.c_str()));

  auto eventCls = env->FindClass("polyregion/Event");
  auto epochMillisField = env->GetFieldID(eventCls, "epochMillis", "J");
  auto nameField = env->GetFieldID(eventCls, "name", StringSignature);
  auto eventNanosField = env->GetFieldID(eventCls, "elapsedNanos", "J");
  auto events = env->NewObjectArray(jint(c.events.size()), eventCls, nullptr);
  for (jsize i = 0; i < jsize(c.events.size()); ++i) {
    auto elem = newNoArgObject(env, eventCls);
    env->SetLongField(elem, epochMillisField, jlong(c.events[i].epochMillis));
    env->SetObjectField(elem, nameField, env->NewStringUTF(c.events[i].name.c_str()));
    env->SetLongField(elem, eventNanosField, jlong(c.events[i].elapsedNanos));
    env->SetObjectArrayElement(events, i, elem);
  }
  env->SetObjectField(compilation, eventsField, events);
  return compilation;
}

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  polyregion::compiler::initialise();
  return JNI_VERSION_1_1;
}