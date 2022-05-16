#include <string>
#include <vector>

#include "compiler.h"
#include "polyregion_PolyregionCompiler.h"
#include "polyregion_compiler.h"

static constexpr const char *StringArraySignature = "[Ljava/lang/String;";
static constexpr const char *StringSignature = "Ljava/lang/String;";
static constexpr const char *LongSignature = "J";

static void throwGeneric(JNIEnv *env, const std::string &message) {
  if (auto exClass = env->FindClass("polyregion/PolyregionCompilerException"); exClass) {
    env->ThrowNew(exClass, message.c_str());
  }
}

static jobject newNoArgObject(JNIEnv *env, jclass clazz) {
  auto ctor = env->GetMethodID(clazz, "<init>", "()V"); // no parameters
  return env->NewObject(clazz, ctor);
}

template <typename T>
static T transformByteArray(JNIEnv *env, jbyteArray data,
                            const std::function<T(const polyregion::compiler::Bytes &)> &f) {
  auto dataBytes = env->GetByteArrayElements(data, nullptr);
  auto bytes = polyregion::compiler::Bytes(dataBytes, dataBytes + env->GetArrayLength(data));
  env->ReleaseByteArrayElements(data, dataBytes, JNI_ABORT);
  return f(bytes);
}

static jobject mkLayout(JNIEnv *env, const polyregion::compiler::Layout &l) {
  auto layoutCls = env->FindClass("polyregion/Layout");
  auto layout = newNoArgObject(env, layoutCls);
  auto nameField = env->GetFieldID(layoutCls, "name", StringArraySignature);
  auto sizeInBytesField = env->GetFieldID(layoutCls, "sizeInBytes", LongSignature);
  auto alignmentField = env->GetFieldID(layoutCls, "alignment", LongSignature);
  auto membersField = env->GetFieldID(layoutCls, "members", "[Lpolyregion/Member;");

  auto nameFqn = env->NewObjectArray(jint(l.name.fqn.size()), env->FindClass("java/lang/String"), nullptr);
  for (jsize i = 0; i < jsize(l.name.fqn.size()); ++i) {
    env->SetObjectArrayElement(nameFqn, i, env->NewStringUTF(l.name.fqn[i].c_str()));
  }
  env->SetObjectField(layout, nameField, nameFqn);
  env->SetLongField(layout, sizeInBytesField, jlong(l.sizeInBytes));
  env->SetLongField(layout, alignmentField, jlong(l.alignment));

  auto memberCls = env->FindClass("polyregion/Member");
  auto memberNameField = env->GetFieldID(memberCls, "name", StringSignature);
  auto memberSizeInBytesField = env->GetFieldID(memberCls, "sizeInBytes", LongSignature);
  auto memberOffsetInBytesField = env->GetFieldID(memberCls, "offsetInBytes", LongSignature);

  auto member = env->NewObjectArray(jint(l.members.size()), memberCls, nullptr);
  for (jsize i = 0; i < jsize(l.members.size()); ++i) {
    auto elem = newNoArgObject(env, memberCls);
    env->SetLongField(elem, memberSizeInBytesField, jlong(l.members[i].sizeInBytes));
    env->SetLongField(elem, memberOffsetInBytesField, jlong(l.members[i].offsetInBytes));
    env->SetObjectField(elem, memberNameField, env->NewStringUTF(l.members[i].name.symbol.c_str()));
    env->SetObjectArrayElement(member, i, elem);
  }
  env->SetObjectField(layout, membersField, member);

  return layout;
}

JNIEXPORT jobject JNICALL Java_polyregion_PolyregionCompiler_layoutOf(JNIEnv *env, jclass thisCls,
                                                                      jbyteArray structDef) {
  auto l = transformByteArray<polyregion::compiler::Layout>(
      env, structDef, [&](auto &&xs) { return polyregion::compiler::layoutOf(xs, <#initializer #>); });
  return mkLayout(env, l);
}

jobject Java_polyregion_PolyregionCompiler_compile(JNIEnv *env, jclass thisCls, jbyteArray function,
                                                   jboolean emitDisassembly, jshort backend) {

  auto c = transformByteArray<polyregion::compiler::Compilation>(
      env, function, [](auto &&xs) { return polyregion::compiler::compile(xs); });

  auto compilationCls = env->FindClass("polyregion/Compilation");
  auto compilation = newNoArgObject(env, compilationCls);
  auto programField = env->GetFieldID(compilationCls, "program", "[B");
  auto messagesField = env->GetFieldID(compilationCls, "messages", StringSignature);
  auto eventsField = env->GetFieldID(compilationCls, "events", "[Lpolyregion/Event;");
  auto layoutsField = env->GetFieldID(compilationCls, "layouts", "[Lpolyregion/Layout;");

  if (c.binary) {
    auto bin = env->NewByteArray(jsize(c.binary->size()));
    auto binElems = env->GetByteArrayElements(bin, nullptr);
    std::copy(c.binary->begin(), c.binary->end(), binElems);
    env->ReleaseByteArrayElements(bin, binElems, JNI_COMMIT);
    env->SetObjectField(compilation, programField, bin);
  } else {
    throwGeneric(env, c.messages);
  }

  env->SetObjectField(compilation, messagesField, env->NewStringUTF(c.messages.c_str()));

  auto layouts = env->NewObjectArray(jint(c.layouts.size()), env->FindClass("polyregion/Layout"), nullptr);
  for (jsize i = 0; i < jsize(c.layouts.size()); ++i) {
    env->SetObjectArrayElement(layouts, i, mkLayout(env, c.layouts[i]));
  }
  env->SetObjectField(compilation, layoutsField, layouts);

  auto eventCls = env->FindClass("polyregion/Event");
  auto epochMillisField = env->GetFieldID(eventCls, "epochMillis", LongSignature);
  auto nameField = env->GetFieldID(eventCls, "name", StringSignature);
  auto dataField = env->GetFieldID(eventCls, "data", StringSignature);
  auto eventNanosField = env->GetFieldID(eventCls, "elapsedNanos", LongSignature);
  auto events = env->NewObjectArray(jint(c.events.size()), eventCls, nullptr);
  for (jsize i = 0; i < jsize(c.events.size()); ++i) {
    auto elem = newNoArgObject(env, eventCls);
    env->SetLongField(elem, epochMillisField, jlong(c.events[i].epochMillis));
    env->SetLongField(elem, eventNanosField, jlong(c.events[i].elapsedNanos));
    env->SetObjectField(elem, nameField, env->NewStringUTF(c.events[i].name.c_str()));
    env->SetObjectField(elem, dataField, env->NewStringUTF(c.events[i].data.c_str()));
    env->SetObjectArrayElement(events, i, elem);
  }
  env->SetObjectField(compilation, eventsField, events);
  return compilation;
}

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  polyregion::compiler::initialise();
  return JNI_VERSION_1_1;
}