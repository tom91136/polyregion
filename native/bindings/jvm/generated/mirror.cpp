#include "mirror.h"
using namespace polyregion::generated;
ByteBuffer::Instance::Instance(const ByteBuffer &meta, jobject instance) : meta(meta), instance(instance) {}
jbyteArray ByteBuffer::Instance::hb(JNIEnv *env) const { return reinterpret_cast<jbyteArray>(env->GetObjectField(instance, meta.hbField)); }
jint ByteBuffer::Instance::offset(JNIEnv *env) const { return env->GetIntField(instance, meta.offsetField); }
jboolean ByteBuffer::Instance::isReadOnly(JNIEnv *env) const { return env->GetBooleanField(instance, meta.isReadOnlyField); }
jboolean ByteBuffer::Instance::bigEndian(JNIEnv *env) const { return env->GetBooleanField(instance, meta.bigEndianField); }
jboolean ByteBuffer::Instance::nativeByteOrder(JNIEnv *env) const { return env->GetBooleanField(instance, meta.nativeByteOrderField); }
ByteBuffer::ByteBuffer(JNIEnv *env)
    : clazz(env->FindClass("java/nio/ByteBuffer")),
      hbField(env->GetFieldID(clazz, "hb", "[B")),
      offsetField(env->GetFieldID(clazz, "offset", "I")),
      isReadOnlyField(env->GetFieldID(clazz, "isReadOnly", "Z")),
      bigEndianField(env->GetFieldID(clazz, "bigEndian", "Z")),
      nativeByteOrderField(env->GetFieldID(clazz, "nativeByteOrder", "Z")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "([BJILjdk/internal/access/foreign/MemorySegmentProxy;)V")),
      ctor1Method(env->GetMethodID(clazz, "<init>", "(IIIILjdk/internal/access/foreign/MemorySegmentProxy;)V")),
      ctor2Method(env->GetMethodID(clazz, "<init>", "(IIII[BILjdk/internal/access/foreign/MemorySegmentProxy;)V")),
      allocate_ILjava_nio_ByteBuffer_Method(env->GetStaticMethodID(clazz, "allocate", "(I)Ljava/nio/ByteBuffer;")),
      allocateDirect_ILjava_nio_ByteBuffer_Method(env->GetStaticMethodID(clazz, "allocateDirect", "(I)Ljava/nio/ByteBuffer;")) {};
ByteBuffer::Instance ByteBuffer::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
jobject ByteBuffer::allocate(JNIEnv *env, jint arg0) const { return env->CallStaticObjectMethod(clazz, allocate_ILjava_nio_ByteBuffer_Method); }
jobject ByteBuffer::allocateDirect(JNIEnv *env, jint arg0) const { return env->CallStaticObjectMethod(clazz, allocateDirect_ILjava_nio_ByteBuffer_Method); }
ByteBuffer::Instance ByteBuffer::operator()(JNIEnv *env, jbyteArray arg0, jlong arg1, jint arg2, jobject arg3) const {
  return {*this, env->NewObject(clazz, ctor0Method, arg0, arg1, arg2, arg3)};
}
ByteBuffer::Instance ByteBuffer::operator()(JNIEnv *env, jint arg0, jint arg1, jint arg2, jint arg3, jobject arg4) const {
  return {*this, env->NewObject(clazz, ctor1Method, arg0, arg1, arg2, arg3, arg4)};
}
ByteBuffer::Instance ByteBuffer::operator()(JNIEnv *env, jint arg0, jint arg1, jint arg2, jint arg3, jbyteArray arg4, jint arg5, jobject arg6) const {
  return {*this, env->NewObject(clazz, ctor2Method, arg0, arg1, arg2, arg3, arg4, arg5, arg6)};
}

Property::Instance::Instance(const Property &meta, jobject instance) : meta(meta), instance(instance) {}
jstring Property::Instance::key(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.keyField)); }
jstring Property::Instance::value(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.valueField)); }
Property::Property(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/runtime/Property")),
      keyField(env->GetFieldID(clazz, "key", "Ljava/lang/String;")),
      valueField(env->GetFieldID(clazz, "value", "Ljava/lang/String;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V")) {};
Property::Instance Property::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Property::Instance Property::operator()(JNIEnv *env, jstring key, jstring value) const {
  return {*this, env->NewObject(clazz, ctor0Method, key, value)};
}

Dim3::Instance::Instance(const Dim3 &meta, jobject instance) : meta(meta), instance(instance) {}
jlong Dim3::Instance::x(JNIEnv *env) const { return env->GetLongField(instance, meta.xField); }
jlong Dim3::Instance::y(JNIEnv *env) const { return env->GetLongField(instance, meta.yField); }
jlong Dim3::Instance::z(JNIEnv *env) const { return env->GetLongField(instance, meta.zField); }
Dim3::Dim3(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/runtime/Dim3")),
      xField(env->GetFieldID(clazz, "x", "J")),
      yField(env->GetFieldID(clazz, "y", "J")),
      zField(env->GetFieldID(clazz, "z", "J")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JJJ)V")) {};
Dim3::Instance Dim3::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Dim3::Instance Dim3::operator()(JNIEnv *env, jlong x, jlong y, jlong z) const {
  return {*this, env->NewObject(clazz, ctor0Method, x, y, z)};
}

Policy::Instance::Instance(const Policy &meta, jobject instance) : meta(meta), instance(instance) {}
Dim3::Instance Policy::Instance::global(JNIEnv *env, const Dim3& clazz_) const { return {clazz_, env->GetObjectField(instance, meta.globalField)}; }
Dim3::Instance Policy::Instance::local(JNIEnv *env, const Dim3& clazz_) const { return {clazz_, env->GetObjectField(instance, meta.localField)}; }
jobject Policy::Instance::local(JNIEnv *env) const { return env->CallObjectMethod(instance, meta.local_Ljava_util_Optional_Method); }
Policy::Policy(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/runtime/Policy")),
      globalField(env->GetFieldID(clazz, "global", "Lpolyregion/jvm/runtime/Dim3;")),
      localField(env->GetFieldID(clazz, "local", "Lpolyregion/jvm/runtime/Dim3;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Lpolyregion/jvm/runtime/Dim3;)V")),
      local_Ljava_util_Optional_Method(env->GetMethodID(clazz, "local", "()Ljava/util/Optional;")) {};
Policy::Instance Policy::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Policy::Instance Policy::operator()(JNIEnv *env, jobject global) const {
  return {*this, env->NewObject(clazz, ctor0Method, global)};
}

Queue::Instance::Instance(const Queue &meta, jobject instance) : meta(meta), instance(instance) {}
jlong Queue::Instance::nativePeer(JNIEnv *env) const { return env->GetLongField(instance, meta.nativePeerField); }
void Queue::Instance::enqueueHostToDeviceAsync(JNIEnv *env, jobject src, jlong dst, jint size, jobject cb) const { env->CallVoidMethod(instance, meta.enqueueHostToDeviceAsync_Ljava_nio_ByteBuffer_JILjava_lang_Runnable_VMethod); }
void Queue::Instance::enqueueInvokeAsync(JNIEnv *env, jstring moduleName, jstring symbol, jbyteArray argTys, jobjectArray argBuffers, jbyte rtnTy, jobject rtnBuffer, jobject policy, jobject cb) const { env->CallVoidMethod(instance, meta.enqueueInvokeAsync_Ljava_lang_String_Ljava_lang_String_aBaLjava_nio_ByteBuffer_BLjava_nio_ByteBuffer_Lpolyregion_jvm_runtime_Policy_Ljava_lang_Runnable_VMethod); }
void Queue::Instance::enqueueInvokeAsync(JNIEnv *env, jstring moduleName, jstring symbol, jobject args, jobject rtn, jobject policy, jobject cb) const { env->CallVoidMethod(instance, meta.enqueueInvokeAsync_Ljava_lang_String_Ljava_lang_String_Ljava_util_List_Lpolyregion_jvm_runtime_Arg_Lpolyregion_jvm_runtime_Policy_Ljava_lang_Runnable_VMethod); }
void Queue::Instance::enqueueDeviceToHostAsync(JNIEnv *env, jlong src, jobject dst, jint size, jobject cb) const { env->CallVoidMethod(instance, meta.enqueueDeviceToHostAsync_JLjava_nio_ByteBuffer_ILjava_lang_Runnable_VMethod); }
Queue::Queue(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/runtime/Device$Queue")),
      nativePeerField(env->GetFieldID(clazz, "nativePeer", "J")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(J)V")),
      enqueueHostToDeviceAsync_Ljava_nio_ByteBuffer_JILjava_lang_Runnable_VMethod(env->GetMethodID(clazz, "enqueueHostToDeviceAsync", "(Ljava/nio/ByteBuffer;JILjava/lang/Runnable;)V")),
      enqueueInvokeAsync_Ljava_lang_String_Ljava_lang_String_aBaLjava_nio_ByteBuffer_BLjava_nio_ByteBuffer_Lpolyregion_jvm_runtime_Policy_Ljava_lang_Runnable_VMethod(env->GetMethodID(clazz, "enqueueInvokeAsync", "(Ljava/lang/String;Ljava/lang/String;[B[Ljava/nio/ByteBuffer;BLjava/nio/ByteBuffer;Lpolyregion/jvm/runtime/Policy;Ljava/lang/Runnable;)V")),
      enqueueInvokeAsync_Ljava_lang_String_Ljava_lang_String_Ljava_util_List_Lpolyregion_jvm_runtime_Arg_Lpolyregion_jvm_runtime_Policy_Ljava_lang_Runnable_VMethod(env->GetMethodID(clazz, "enqueueInvokeAsync", "(Ljava/lang/String;Ljava/lang/String;Ljava/util/List;Lpolyregion/jvm/runtime/Arg;Lpolyregion/jvm/runtime/Policy;Ljava/lang/Runnable;)V")),
      enqueueDeviceToHostAsync_JLjava_nio_ByteBuffer_ILjava_lang_Runnable_VMethod(env->GetMethodID(clazz, "enqueueDeviceToHostAsync", "(JLjava/nio/ByteBuffer;ILjava/lang/Runnable;)V")) {};
Queue::Instance Queue::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Queue::Instance Queue::operator()(JNIEnv *env, jlong nativePeer) const {
  return {*this, env->NewObject(clazz, ctor0Method, nativePeer)};
}

Device::Instance::Instance(const Device &meta, jobject instance) : meta(meta), instance(instance) {}
jlong Device::Instance::nativePeer(JNIEnv *env) const { return env->GetLongField(instance, meta.nativePeerField); }
jlong Device::Instance::id(JNIEnv *env) const { return env->GetLongField(instance, meta.idField); }
jstring Device::Instance::name(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.nameField)); }
jobjectArray Device::Instance::properties(JNIEnv *env) const { return reinterpret_cast<jobjectArray>(env->GetObjectField(instance, meta.propertiesField)); }
jlong Device::Instance::malloc(JNIEnv *env, jlong size, jobject access) const { return env->CallLongMethod(instance, meta.malloc_JLpolyregion_jvm_runtime_Access_JMethod); }
Queue::Instance Device::Instance::createQueue(JNIEnv *env, const Queue& clazz_) const { return {clazz_, env->CallObjectMethod(instance, meta.createQueue_Lpolyregion_jvm_runtime_Device_Queue_Method)}; }
void Device::Instance::loadModule(JNIEnv *env, jstring name, jbyteArray image) const { env->CallVoidMethod(instance, meta.loadModule_Ljava_lang_String_aBVMethod); }
void Device::Instance::free(JNIEnv *env, jlong data) const { env->CallVoidMethod(instance, meta.free_JVMethod); }
Device::Device(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/runtime/Device")),
      nativePeerField(env->GetFieldID(clazz, "nativePeer", "J")),
      idField(env->GetFieldID(clazz, "id", "J")),
      nameField(env->GetFieldID(clazz, "name", "Ljava/lang/String;")),
      propertiesField(env->GetFieldID(clazz, "properties", "[Lpolyregion/jvm/runtime/Property;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JJLjava/lang/String;[Lpolyregion/jvm/runtime/Property;)V")),
      malloc_JLpolyregion_jvm_runtime_Access_JMethod(env->GetMethodID(clazz, "malloc", "(JLpolyregion/jvm/runtime/Access;)J")),
      createQueue_Lpolyregion_jvm_runtime_Device_Queue_Method(env->GetMethodID(clazz, "createQueue", "()Lpolyregion/jvm/runtime/Device$Queue;")),
      loadModule_Ljava_lang_String_aBVMethod(env->GetMethodID(clazz, "loadModule", "(Ljava/lang/String;[B)V")),
      free_JVMethod(env->GetMethodID(clazz, "free", "(J)V")) {};
Device::Instance Device::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Device::Instance Device::operator()(JNIEnv *env, jlong nativePeer, jlong id, jstring name, jobjectArray properties) const {
  return {*this, env->NewObject(clazz, ctor0Method, nativePeer, id, name, properties)};
}

Runtime::Instance::Instance(const Runtime &meta, jobject instance) : meta(meta), instance(instance) {}
jlong Runtime::Instance::nativePeer(JNIEnv *env) const { return env->GetLongField(instance, meta.nativePeerField); }
jstring Runtime::Instance::name(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.nameField)); }
jobjectArray Runtime::Instance::properties(JNIEnv *env) const { return reinterpret_cast<jobjectArray>(env->GetObjectField(instance, meta.propertiesField)); }
jobjectArray Runtime::Instance::devices(JNIEnv *env) const { return reinterpret_cast<jobjectArray>(env->CallObjectMethod(instance, meta.devices_aLpolyregion_jvm_runtime_Device_Method)); }
Runtime::Runtime(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/runtime/Runtime")),
      nativePeerField(env->GetFieldID(clazz, "nativePeer", "J")),
      nameField(env->GetFieldID(clazz, "name", "Ljava/lang/String;")),
      propertiesField(env->GetFieldID(clazz, "properties", "[Lpolyregion/jvm/runtime/Property;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JLjava/lang/String;[Lpolyregion/jvm/runtime/Property;)V")),
      devices_aLpolyregion_jvm_runtime_Device_Method(env->GetMethodID(clazz, "devices", "()[Lpolyregion/jvm/runtime/Device;")) {};
Runtime::Instance Runtime::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Runtime::Instance Runtime::operator()(JNIEnv *env, jlong nativePeer, jstring name, jobjectArray properties) const {
  return {*this, env->NewObject(clazz, ctor0Method, nativePeer, name, properties)};
}

Event::Instance::Instance(const Event &meta, jobject instance) : meta(meta), instance(instance) {}
jlong Event::Instance::epochMillis(JNIEnv *env) const { return env->GetLongField(instance, meta.epochMillisField); }
jlong Event::Instance::elapsedNanos(JNIEnv *env) const { return env->GetLongField(instance, meta.elapsedNanosField); }
jstring Event::Instance::name(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.nameField)); }
jstring Event::Instance::data(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.dataField)); }
Event::Event(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/compiler/Event")),
      epochMillisField(env->GetFieldID(clazz, "epochMillis", "J")),
      elapsedNanosField(env->GetFieldID(clazz, "elapsedNanos", "J")),
      nameField(env->GetFieldID(clazz, "name", "Ljava/lang/String;")),
      dataField(env->GetFieldID(clazz, "data", "Ljava/lang/String;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JJLjava/lang/String;Ljava/lang/String;)V")) {};
Event::Instance Event::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Event::Instance Event::operator()(JNIEnv *env, jlong epochMillis, jlong elapsedNanos, jstring name, jstring data) const {
  return {*this, env->NewObject(clazz, ctor0Method, epochMillis, elapsedNanos, name, data)};
}

Layout::Instance::Instance(const Layout &meta, jobject instance) : meta(meta), instance(instance) {}
jobjectArray Layout::Instance::name(JNIEnv *env) const { return reinterpret_cast<jobjectArray>(env->GetObjectField(instance, meta.nameField)); }
jlong Layout::Instance::sizeInBytes(JNIEnv *env) const { return env->GetLongField(instance, meta.sizeInBytesField); }
jlong Layout::Instance::alignment(JNIEnv *env) const { return env->GetLongField(instance, meta.alignmentField); }
jobjectArray Layout::Instance::members(JNIEnv *env) const { return reinterpret_cast<jobjectArray>(env->GetObjectField(instance, meta.membersField)); }
Layout::Layout(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/compiler/Layout")),
      nameField(env->GetFieldID(clazz, "name", "[Ljava/lang/String;")),
      sizeInBytesField(env->GetFieldID(clazz, "sizeInBytes", "J")),
      alignmentField(env->GetFieldID(clazz, "alignment", "J")),
      membersField(env->GetFieldID(clazz, "members", "[Lpolyregion/jvm/compiler/Member;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "([Ljava/lang/String;JJ[Lpolyregion/jvm/compiler/Member;)V")) {};
Layout::Instance Layout::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Layout::Instance Layout::operator()(JNIEnv *env, jobjectArray name, jlong sizeInBytes, jlong alignment, jobjectArray members) const {
  return {*this, env->NewObject(clazz, ctor0Method, name, sizeInBytes, alignment, members)};
}

Member::Instance::Instance(const Member &meta, jobject instance) : meta(meta), instance(instance) {}
jstring Member::Instance::name(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.nameField)); }
jlong Member::Instance::offsetInBytes(JNIEnv *env) const { return env->GetLongField(instance, meta.offsetInBytesField); }
jlong Member::Instance::sizeInBytes(JNIEnv *env) const { return env->GetLongField(instance, meta.sizeInBytesField); }
Member::Member(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/compiler/Member")),
      nameField(env->GetFieldID(clazz, "name", "Ljava/lang/String;")),
      offsetInBytesField(env->GetFieldID(clazz, "offsetInBytes", "J")),
      sizeInBytesField(env->GetFieldID(clazz, "sizeInBytes", "J")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;JJ)V")) {};
Member::Instance Member::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Member::Instance Member::operator()(JNIEnv *env, jstring name, jlong offsetInBytes, jlong sizeInBytes) const {
  return {*this, env->NewObject(clazz, ctor0Method, name, offsetInBytes, sizeInBytes)};
}

Options::Instance::Instance(const Options &meta, jobject instance) : meta(meta), instance(instance) {}
jbyte Options::Instance::target(JNIEnv *env) const { return env->GetByteField(instance, meta.targetField); }
jstring Options::Instance::arch(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.archField)); }
Options::Options(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/compiler/Options")),
      targetField(env->GetFieldID(clazz, "target", "B")),
      archField(env->GetFieldID(clazz, "arch", "Ljava/lang/String;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(BLjava/lang/String;)V")) {};
Options::Instance Options::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Options::Instance Options::operator()(JNIEnv *env, jbyte target, jstring arch) const {
  return {*this, env->NewObject(clazz, ctor0Method, target, arch)};
}

Compilation::Instance::Instance(const Compilation &meta, jobject instance) : meta(meta), instance(instance) {}
jbyteArray Compilation::Instance::program(JNIEnv *env) const { return reinterpret_cast<jbyteArray>(env->GetObjectField(instance, meta.programField)); }
jobjectArray Compilation::Instance::events(JNIEnv *env) const { return reinterpret_cast<jobjectArray>(env->GetObjectField(instance, meta.eventsField)); }
jobjectArray Compilation::Instance::layouts(JNIEnv *env) const { return reinterpret_cast<jobjectArray>(env->GetObjectField(instance, meta.layoutsField)); }
jstring Compilation::Instance::messages(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.messagesField)); }
Compilation::Compilation(JNIEnv *env)
    : clazz(env->FindClass("polyregion/jvm/compiler/Compilation")),
      programField(env->GetFieldID(clazz, "program", "[B")),
      eventsField(env->GetFieldID(clazz, "events", "[Lpolyregion/jvm/compiler/Event;")),
      layoutsField(env->GetFieldID(clazz, "layouts", "[Lpolyregion/jvm/compiler/Layout;")),
      messagesField(env->GetFieldID(clazz, "messages", "Ljava/lang/String;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "([B[Lpolyregion/jvm/compiler/Event;[Lpolyregion/jvm/compiler/Layout;Ljava/lang/String;)V")) {};
Compilation::Instance Compilation::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Compilation::Instance Compilation::operator()(JNIEnv *env, jbyteArray program, jobjectArray events, jobjectArray layouts, jstring messages) const {
  return {*this, env->NewObject(clazz, ctor0Method, program, events, layouts, messages)};
}

Runnable::Instance::Instance(const Runnable &meta, jobject instance) : meta(meta), instance(instance) {}
void Runnable::Instance::run(JNIEnv *env) const { env->CallVoidMethod(instance, meta.run_VMethod); }
Runnable::Runnable(JNIEnv *env)
    : clazz(env->FindClass("java/lang/Runnable")),
      run_VMethod(env->GetMethodID(clazz, "run", "()V")) {};
Runnable::Instance Runnable::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }


