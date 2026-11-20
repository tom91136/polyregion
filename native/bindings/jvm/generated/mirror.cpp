#include "mirror.h"
using namespace polyregion::generated;
ByteBuffer::Instance::Instance(const ByteBuffer &meta, jobject instance) : meta(meta), instance(instance) {}

ByteBuffer::ByteBuffer(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))),
      allocateDirect_ILjava_nio_ByteBuffer_Method(env->GetStaticMethodID(clazz, "allocateDirect", "(I)Ljava/nio/ByteBuffer;")),
      allocate_ILjava_nio_ByteBuffer_Method(env->GetStaticMethodID(clazz, "allocate", "(I)Ljava/nio/ByteBuffer;")) {};
thread_local std::vector<std::unique_ptr<ByteBuffer>> ByteBuffer::cached = {};
ByteBuffer &ByteBuffer::of(JNIEnv *env) {
  jclass found = env->FindClass("java/nio/ByteBuffer");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new ByteBuffer(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void ByteBuffer::drop(JNIEnv *env) {
  jclass found = env->FindClass("java/nio/ByteBuffer");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
ByteBuffer::Instance ByteBuffer::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
jobject ByteBuffer::allocateDirect(JNIEnv *env, jint arg0) const {
  return env->CallStaticObjectMethod(clazz, allocateDirect_ILjava_nio_ByteBuffer_Method, arg0);
}
jobject ByteBuffer::allocate(JNIEnv *env, jint arg0) const {
  return env->CallStaticObjectMethod(clazz, allocate_ILjava_nio_ByteBuffer_Method, arg0);
}

Property::Instance::Instance(const Property &meta, jobject instance) : meta(meta), instance(instance) {}

Property::Property(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V")) {};
thread_local std::vector<std::unique_ptr<Property>> Property::cached = {};
Property &Property::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Property");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Property(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Property::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Property");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Property::Instance Property::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Property::Instance Property::operator()(JNIEnv *env, jstring key, jstring value) const {
  return {*this, env->NewObject(clazz, ctor0Method, key, value)};
}

Dim3::Instance::Instance(const Dim3 &meta, jobject instance) : meta(meta), instance(instance) {}
jlong Dim3::Instance::x(JNIEnv *env) const { return env->GetLongField(instance, meta.xField); }
jlong Dim3::Instance::y(JNIEnv *env) const { return env->GetLongField(instance, meta.yField); }
jlong Dim3::Instance::z(JNIEnv *env) const { return env->GetLongField(instance, meta.zField); }
Dim3::Dim3(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))), xField(env->GetFieldID(clazz, "x", "J")),
      yField(env->GetFieldID(clazz, "y", "J")), zField(env->GetFieldID(clazz, "z", "J")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JJJ)V")) {};
thread_local std::vector<std::unique_ptr<Dim3>> Dim3::cached = {};
Dim3 &Dim3::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Dim3");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Dim3(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Dim3::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Dim3");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Dim3::Instance Dim3::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Dim3::Instance Dim3::operator()(JNIEnv *env, jlong x, jlong y, jlong z) const {
  return {*this, env->NewObject(clazz, ctor0Method, x, y, z)};
}

Policy::Instance::Instance(const Policy &meta, jobject instance) : meta(meta), instance(instance) {}
Dim3::Instance Policy::Instance::global(JNIEnv *env, const Dim3 &clazz_) const {
  return {clazz_, env->GetObjectField(instance, meta.globalField)};
}
Dim3::Instance Policy::Instance::local(JNIEnv *env, const Dim3 &clazz_) const {
  return {clazz_, env->GetObjectField(instance, meta.localField)};
}
jint Policy::Instance::localMemoryBytes(JNIEnv *env) const { return env->GetIntField(instance, meta.localMemoryBytesField); }
Policy::Policy(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))),
      globalField(env->GetFieldID(clazz, "global", "Lpolyregion/jvm/runtime/Dim3;")),
      localField(env->GetFieldID(clazz, "local", "Lpolyregion/jvm/runtime/Dim3;")),
      localMemoryBytesField(env->GetFieldID(clazz, "localMemoryBytes", "I")),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Lpolyregion/jvm/runtime/Dim3;)V")),
      ctor1Method(env->GetMethodID(clazz, "<init>", "(Lpolyregion/jvm/runtime/Dim3;Lpolyregion/jvm/runtime/Dim3;)V")),
      ctor2Method(env->GetMethodID(clazz, "<init>", "(Lpolyregion/jvm/runtime/Dim3;Lpolyregion/jvm/runtime/Dim3;I)V")) {};
thread_local std::vector<std::unique_ptr<Policy>> Policy::cached = {};
Policy &Policy::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Policy");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Policy(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Policy::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Policy");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Policy::Instance Policy::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Policy::Instance Policy::operator()(JNIEnv *env, jobject global) const { return {*this, env->NewObject(clazz, ctor0Method, global)}; }
Policy::Instance Policy::operator()(JNIEnv *env, jobject global, jobject local) const {
  return {*this, env->NewObject(clazz, ctor1Method, global, local)};
}
Policy::Instance Policy::operator()(JNIEnv *env, jobject global, jobject local, jint localMemoryBytes) const {
  return {*this, env->NewObject(clazz, ctor2Method, global, local, localMemoryBytes)};
}

Device::Instance::Instance(const Device &meta, jobject instance) : meta(meta), instance(instance) {}

Device::Device(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JJLjava/lang/String;Z)V")) {};
thread_local std::vector<std::unique_ptr<Device>> Device::cached = {};
Device &Device::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Device");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Device(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Device::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Device");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Device::Instance Device::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Device::Instance Device::operator()(JNIEnv *env, jlong nativePeer, jlong id, jstring name, jboolean sharedAddressSpace) const {
  return {*this, env->NewObject(clazz, ctor0Method, nativePeer, id, name, sharedAddressSpace)};
}

Platform::Instance::Instance(const Platform &meta, jobject instance) : meta(meta), instance(instance) {}

Platform::Platform(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JLjava/lang/String;)V")) {};
thread_local std::vector<std::unique_ptr<Platform>> Platform::cached = {};
Platform &Platform::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Platform");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Platform(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Platform::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/runtime/Platform");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Platform::Instance Platform::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Platform::Instance Platform::operator()(JNIEnv *env, jlong nativePeer, jstring name) const {
  return {*this, env->NewObject(clazz, ctor0Method, nativePeer, name)};
}

Event::Instance::Instance(const Event &meta, jobject instance) : meta(meta), instance(instance) {}

Event::Event(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(JJLjava/lang/String;Ljava/lang/String;[Lpolyregion/jvm/compiler/Event;)V")) {};
thread_local std::vector<std::unique_ptr<Event>> Event::cached = {};
Event &Event::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Event");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Event(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Event::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Event");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Event::Instance Event::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Event::Instance Event::operator()(JNIEnv *env, jlong epochMillis, jlong elapsedNanos, jstring name, jstring data,
                                  jobjectArray items) const {
  return {*this, env->NewObject(clazz, ctor0Method, epochMillis, elapsedNanos, name, data, items)};
}

Layout::Instance::Instance(const Layout &meta, jobject instance) : meta(meta), instance(instance) {}

Layout::Layout(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;JJ[Lpolyregion/jvm/compiler/Layout$Member;)V")) {};
thread_local std::vector<std::unique_ptr<Layout>> Layout::cached = {};
Layout &Layout::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Layout");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Layout(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Layout::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Layout");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Layout::Instance Layout::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Layout::Instance Layout::operator()(JNIEnv *env, jstring name, jlong sizeInBytes, jlong alignment, jobjectArray members) const {
  return {*this, env->NewObject(clazz, ctor0Method, name, sizeInBytes, alignment, members)};
}

Member::Instance::Instance(const Member &meta, jobject instance) : meta(meta), instance(instance) {}

Member::Member(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))),
      ctor0Method(env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;JJ)V")) {};
thread_local std::vector<std::unique_ptr<Member>> Member::cached = {};
Member &Member::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Layout$Member");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Member(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Member::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Layout$Member");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Member::Instance Member::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Member::Instance Member::operator()(JNIEnv *env, jstring name, jlong offsetInBytes, jlong sizeInBytes) const {
  return {*this, env->NewObject(clazz, ctor0Method, name, offsetInBytes, sizeInBytes)};
}

Options::Instance::Instance(const Options &meta, jobject instance) : meta(meta), instance(instance) {}
jstring Options::Instance::arch(JNIEnv *env) const { return reinterpret_cast<jstring>(env->GetObjectField(instance, meta.archField)); }
jbyte Options::Instance::target(JNIEnv *env) const { return env->GetByteField(instance, meta.targetField); }
Options::Options(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))), archField(env->GetFieldID(clazz, "arch", "Ljava/lang/String;")),
      targetField(env->GetFieldID(clazz, "target", "B")), ctor0Method(env->GetMethodID(clazz, "<init>", "(BLjava/lang/String;)V")) {};
thread_local std::vector<std::unique_ptr<Options>> Options::cached = {};
Options &Options::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Options");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Options(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Options::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Options");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Options::Instance Options::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Options::Instance Options::operator()(JNIEnv *env, jbyte target, jstring arch) const {
  return {*this, env->NewObject(clazz, ctor0Method, target, arch)};
}

Compilation::Instance::Instance(const Compilation &meta, jobject instance) : meta(meta), instance(instance) {}

Compilation::Compilation(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))),
      ctor0Method(env->GetMethodID(
          clazz, "<init>", "([B[Ljava/lang/String;[Lpolyregion/jvm/compiler/Event;[Lpolyregion/jvm/compiler/Layout;Ljava/lang/String;)V")) {
      };
thread_local std::vector<std::unique_ptr<Compilation>> Compilation::cached = {};
Compilation &Compilation::of(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Compilation");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Compilation(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Compilation::drop(JNIEnv *env) {
  jclass found = env->FindClass("polyregion/jvm/compiler/Compilation");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Compilation::Instance Compilation::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
Compilation::Instance Compilation::operator()(JNIEnv *env, jbyteArray program, jobjectArray features, jobjectArray events,
                                              jobjectArray layouts, jstring messages) const {
  return {*this, env->NewObject(clazz, ctor0Method, program, features, events, layouts, messages)};
}

String::Instance::Instance(const String &meta, jobject instance) : meta(meta), instance(instance) {}

String::String(JNIEnv *env, jclass found) : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))) {};
thread_local std::vector<std::unique_ptr<String>> String::cached = {};
String &String::of(JNIEnv *env) {
  jclass found = env->FindClass("java/lang/String");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new String(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void String::drop(JNIEnv *env) {
  jclass found = env->FindClass("java/lang/String");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
String::Instance String::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }

Runnable::Instance::Instance(const Runnable &meta, jobject instance) : meta(meta), instance(instance) {}
void Runnable::Instance::run(JNIEnv *env) const { env->CallVoidMethod(instance, meta.run_VMethod); }
Runnable::Runnable(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))), run_VMethod(env->GetMethodID(clazz, "run", "()V")) {};
thread_local std::vector<std::unique_ptr<Runnable>> Runnable::cached = {};
Runnable &Runnable::of(JNIEnv *env) {
  jclass found = env->FindClass("java/lang/Runnable");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new Runnable(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void Runnable::drop(JNIEnv *env) {
  jclass found = env->FindClass("java/lang/Runnable");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
Runnable::Instance Runnable::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }

File::Instance::Instance(const File &meta, jobject instance) : meta(meta), instance(instance) {}
jboolean File::Instance::delete_(JNIEnv *env) const { return env->CallBooleanMethod(instance, meta.delete_ZMethod); }
File::File(JNIEnv *env, jclass found)
    : clazz(reinterpret_cast<jclass>(env->NewGlobalRef(found))), delete_ZMethod(env->GetMethodID(clazz, "delete", "()Z")) {};
thread_local std::vector<std::unique_ptr<File>> File::cached = {};
File &File::of(JNIEnv *env) {
  jclass found = env->FindClass("java/io/File");
  for (auto &entry : cached) {
    if (env->IsSameObject(entry->clazz, found)) {
      env->DeleteLocalRef(found);
      return *entry;
    }
  }
  auto &entry = cached.emplace_back(new File(env, found));
  env->DeleteLocalRef(found);
  return *entry;
}
void File::drop(JNIEnv *env) {
  jclass found = env->FindClass("java/io/File");
  if (!found) return env->ExceptionClear();
  std::erase_if(cached, [&](auto &entry) {
    if (!env->IsSameObject(entry->clazz, found)) return false;
    env->DeleteGlobalRef(entry->clazz);
    return true;
  });
  env->DeleteLocalRef(found);
}
File::Instance File::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
