#include "mirror.h"
using namespace polyregion::generated;
ByteBuffer::Instance::Instance(const ByteBuffer &meta, jobject instance) : meta(meta), instance(instance) {}

ByteBuffer::ByteBuffer(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("java/nio/ByteBuffer")))),
      allocate_ILjava_nio_ByteBuffer_Method(env->GetStaticMethodID(clazz, "allocate", "(I)Ljava/nio/ByteBuffer;")),
      allocateDirect_ILjava_nio_ByteBuffer_Method(env->GetStaticMethodID(clazz, "allocateDirect", "(I)Ljava/nio/ByteBuffer;")) { };
thread_local std::unique_ptr<ByteBuffer> ByteBuffer::cached = {};
ByteBuffer& ByteBuffer::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<ByteBuffer>(new ByteBuffer(env));
  return *cached;
}
void ByteBuffer::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
ByteBuffer::Instance ByteBuffer::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
jobject ByteBuffer::allocate(JNIEnv *env, jint arg0) const { return env->CallStaticObjectMethod(clazz, allocate_ILjava_nio_ByteBuffer_Method, arg0); }
jobject ByteBuffer::allocateDirect(JNIEnv *env, jint arg0) const { return env->CallStaticObjectMethod(clazz, allocateDirect_ILjava_nio_ByteBuffer_Method, arg0); }

Property::Instance::Instance(const Property &meta, jobject instance) : meta(meta), instance(instance) {}

Property::Property(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Property")))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V")) { };
thread_local std::unique_ptr<Property> Property::cached = {};
Property& Property::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Property>(new Property(env));
  return *cached;
}
void Property::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Property::Instance Property::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Property::Instance Property::operator()(JNIEnv *env, jstring key, jstring value) const {
  return {*this, env->NewObject(clazz, ctor0Method, key, value)};
}

Dim3::Instance::Instance(const Dim3 &meta, jobject instance) : meta(meta), instance(instance) {}
jlong Dim3::Instance::x(JNIEnv *env) const { return env->GetLongField(instance, meta.xField); }
jlong Dim3::Instance::y(JNIEnv *env) const { return env->GetLongField(instance, meta.yField); }
jlong Dim3::Instance::z(JNIEnv *env) const { return env->GetLongField(instance, meta.zField); }
Dim3::Dim3(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Dim3")))),
      xField(env->GetFieldID(clazz, "x", "J")),
      yField(env->GetFieldID(clazz, "y", "J")),
      zField(env->GetFieldID(clazz, "z", "J")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JJJ)V")) { };
thread_local std::unique_ptr<Dim3> Dim3::cached = {};
Dim3& Dim3::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Dim3>(new Dim3(env));
  return *cached;
}
void Dim3::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Dim3::Instance Dim3::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Dim3::Instance Dim3::operator()(JNIEnv *env, jlong x, jlong y, jlong z) const {
  return {*this, env->NewObject(clazz, ctor0Method, x, y, z)};
}

Policy::Instance::Instance(const Policy &meta, jobject instance) : meta(meta), instance(instance) {}
Dim3::Instance Policy::Instance::global(JNIEnv *env, const Dim3& clazz_) const { return {clazz_, env->GetObjectField(instance, meta.globalField)}; }
Dim3::Instance Policy::Instance::local(JNIEnv *env, const Dim3& clazz_) const { return {clazz_, env->GetObjectField(instance, meta.localField)}; }
Policy::Policy(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Policy")))),
      globalField(env->GetFieldID(clazz, "global", "Lpolyregion/jvm/runtime/Dim3;")),
      localField(env->GetFieldID(clazz, "local", "Lpolyregion/jvm/runtime/Dim3;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Lpolyregion/jvm/runtime/Dim3;)V")),
      ctor1Method(env->GetMethodID(clazz, "<init>", "(Lpolyregion/jvm/runtime/Dim3;Lpolyregion/jvm/runtime/Dim3;)V")) { };
thread_local std::unique_ptr<Policy> Policy::cached = {};
Policy& Policy::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Policy>(new Policy(env));
  return *cached;
}
void Policy::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Policy::Instance Policy::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Policy::Instance Policy::operator()(JNIEnv *env, jobject global) const {
  return {*this, env->NewObject(clazz, ctor0Method, global)};
}
Policy::Instance Policy::operator()(JNIEnv *env, jobject global, jobject local) const {
  return {*this, env->NewObject(clazz, ctor1Method, global, local)};
}

Queue::Instance::Instance(const Queue &meta, jobject instance) : meta(meta), instance(instance) {}

Queue::Queue(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Device$Queue")))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JLpolyregion/jvm/runtime/Device;)V")) { };
thread_local std::unique_ptr<Queue> Queue::cached = {};
Queue& Queue::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Queue>(new Queue(env));
  return *cached;
}
void Queue::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Queue::Instance Queue::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Queue::Instance Queue::operator()(JNIEnv *env, jlong nativePeer, jobject device) const {
  return {*this, env->NewObject(clazz, ctor0Method, nativePeer, device)};
}

Device::Instance::Instance(const Device &meta, jobject instance) : meta(meta), instance(instance) {}

Device::Device(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Device")))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JJLjava/lang/String;Z)V")) { };
thread_local std::unique_ptr<Device> Device::cached = {};
Device& Device::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Device>(new Device(env));
  return *cached;
}
void Device::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Device::Instance Device::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Device::Instance Device::operator()(JNIEnv *env, jlong nativePeer, jlong id, jstring name, jboolean sharedAddressSpace) const {
  return {*this, env->NewObject(clazz, ctor0Method, nativePeer, id, name, sharedAddressSpace)};
}

Runtime::Instance::Instance(const Runtime &meta, jobject instance) : meta(meta), instance(instance) {}

Runtime::Runtime(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/runtime/Runtime")))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JLjava/lang/String;)V")) { };
thread_local std::unique_ptr<Runtime> Runtime::cached = {};
Runtime& Runtime::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Runtime>(new Runtime(env));
  return *cached;
}
void Runtime::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Runtime::Instance Runtime::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Runtime::Instance Runtime::operator()(JNIEnv *env, jlong nativePeer, jstring name) const {
  return {*this, env->NewObject(clazz, ctor0Method, nativePeer, name)};
}

Event::Instance::Instance(const Event &meta, jobject instance) : meta(meta), instance(instance) {}

Event::Event(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/compiler/Event")))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JJLjava/lang/String;Ljava/lang/String;)V")) { };
thread_local std::unique_ptr<Event> Event::cached = {};
Event& Event::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Event>(new Event(env));
  return *cached;
}
void Event::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Event::Instance Event::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Event::Instance Event::operator()(JNIEnv *env, jlong epochMillis, jlong elapsedNanos, jstring name, jstring data) const {
  return {*this, env->NewObject(clazz, ctor0Method, epochMillis, elapsedNanos, name, data)};
}

Layout::Instance::Instance(const Layout &meta, jobject instance) : meta(meta), instance(instance) {}

Layout::Layout(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/compiler/Layout")))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "([Ljava/lang/String;JJ[Lpolyregion/jvm/compiler/Member;)V")) { };
thread_local std::unique_ptr<Layout> Layout::cached = {};
Layout& Layout::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Layout>(new Layout(env));
  return *cached;
}
void Layout::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Layout::Instance Layout::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Layout::Instance Layout::operator()(JNIEnv *env, jobjectArray name, jlong sizeInBytes, jlong alignment, jobjectArray members) const {
  return {*this, env->NewObject(clazz, ctor0Method, name, sizeInBytes, alignment, members)};
}

Member::Instance::Instance(const Member &meta, jobject instance) : meta(meta), instance(instance) {}

Member::Member(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/compiler/Member")))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;JJ)V")) { };
thread_local std::unique_ptr<Member> Member::cached = {};
Member& Member::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Member>(new Member(env));
  return *cached;
}
void Member::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Member::Instance Member::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Member::Instance Member::operator()(JNIEnv *env, jstring name, jlong offsetInBytes, jlong sizeInBytes) const {
  return {*this, env->NewObject(clazz, ctor0Method, name, offsetInBytes, sizeInBytes)};
}

Options::Instance::Instance(const Options &meta, jobject instance) : meta(meta), instance(instance) {}
jbyte Options::Instance::target(JNIEnv *env) const { return env->GetByteField(instance, meta.targetField); }
jstring Options::Instance::arch(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.archField)); }
Options::Options(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/compiler/Options")))),
      targetField(env->GetFieldID(clazz, "target", "B")),
      archField(env->GetFieldID(clazz, "arch", "Ljava/lang/String;")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(BLjava/lang/String;)V")) { };
thread_local std::unique_ptr<Options> Options::cached = {};
Options& Options::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Options>(new Options(env));
  return *cached;
}
void Options::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Options::Instance Options::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Options::Instance Options::operator()(JNIEnv *env, jbyte target, jstring arch) const {
  return {*this, env->NewObject(clazz, ctor0Method, target, arch)};
}

Compilation::Instance::Instance(const Compilation &meta, jobject instance) : meta(meta), instance(instance) {}

Compilation::Compilation(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("polyregion/jvm/compiler/Compilation")))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "([B[Lpolyregion/jvm/compiler/Event;[Lpolyregion/jvm/compiler/Layout;Ljava/lang/String;)V")) { };
thread_local std::unique_ptr<Compilation> Compilation::cached = {};
Compilation& Compilation::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Compilation>(new Compilation(env));
  return *cached;
}
void Compilation::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Compilation::Instance Compilation::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Compilation::Instance Compilation::operator()(JNIEnv *env, jbyteArray program, jobjectArray events, jobjectArray layouts, jstring messages) const {
  return {*this, env->NewObject(clazz, ctor0Method, program, events, layouts, messages)};
}

String::Instance::Instance(const String &meta, jobject instance) : meta(meta), instance(instance) {}

String::String(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("java/lang/String")))) { };
thread_local std::unique_ptr<String> String::cached = {};
String& String::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<String>(new String(env));
  return *cached;
}
void String::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
String::Instance String::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }


Runnable::Instance::Instance(const Runnable &meta, jobject instance) : meta(meta), instance(instance) {}
void Runnable::Instance::run(JNIEnv *env) const { env->CallVoidMethod(instance, meta.run_VMethod); }
Runnable::Runnable(JNIEnv *env)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("java/lang/Runnable")))),
      run_VMethod(env->GetMethodID(clazz, "run", "()V")) { };
thread_local std::unique_ptr<Runnable> Runnable::cached = {};
Runnable& Runnable::of(JNIEnv *env) {
  if(!cached) cached = std::unique_ptr<Runnable>(new Runnable(env));
  return *cached;
}
void Runnable::drop(JNIEnv *env){
  if(cached) {
    env->DeleteGlobalRef(cached->clazz);
    cached.reset();
  }
}
Runnable::Instance Runnable::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }


