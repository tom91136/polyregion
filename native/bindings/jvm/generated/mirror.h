#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <jni.h>
namespace polyregion::generated {
struct ByteBuffer {
  struct Instance {
    const ByteBuffer &meta;
    jobject instance;
    Instance(const ByteBuffer &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
  };
  jclass clazz;
  jmethodID allocateDirect_ILjava_nio_ByteBuffer_Method;
  jmethodID allocate_ILjava_nio_ByteBuffer_Method;

private:
  ByteBuffer(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<ByteBuffer>> cached;

public:
  static ByteBuffer &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  jobject allocateDirect(JNIEnv *env, jint arg0) const;
  jobject allocate(JNIEnv *env, jint arg0) const;
};
struct Property {
  struct Instance {
    const Property &meta;
    jobject instance;
    Instance(const Property &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
  };
  jclass clazz;
  jmethodID ctor0Method;

private:
  Property(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Property>> cached;

public:
  static Property &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jstring key, jstring value) const;
};
struct Dim3 {
  struct Instance {
    const Dim3 &meta;
    jobject instance;
    Instance(const Dim3 &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jlong x(JNIEnv *env) const;
    jlong y(JNIEnv *env) const;
    jlong z(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID xField;
  jfieldID yField;
  jfieldID zField;
  jmethodID ctor0Method;

private:
  Dim3(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Dim3>> cached;

public:
  static Dim3 &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong x, jlong y, jlong z) const;
};
struct Policy {
  struct Instance {
    const Policy &meta;
    jobject instance;
    Instance(const Policy &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    Dim3::Instance global(JNIEnv *env, const Dim3 &clazz) const;
    Dim3::Instance local(JNIEnv *env, const Dim3 &clazz) const;
    jint localMemoryBytes(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID globalField;
  jfieldID localField;
  jfieldID localMemoryBytesField;
  jmethodID ctor0Method;
  jmethodID ctor1Method;
  jmethodID ctor2Method;

private:
  Policy(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Policy>> cached;

public:
  static Policy &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jobject global) const;
  Instance operator()(JNIEnv *env, jobject global, jobject local) const;
  Instance operator()(JNIEnv *env, jobject global, jobject local, jint localMemoryBytes) const;
};
struct Device {
  struct Instance {
    const Device &meta;
    jobject instance;
    Instance(const Device &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
  };
  jclass clazz;
  jmethodID ctor0Method;

private:
  Device(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Device>> cached;

public:
  static Device &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong nativePeer, jlong id, jstring name, jboolean sharedAddressSpace) const;
};
struct Platform {
  struct Instance {
    const Platform &meta;
    jobject instance;
    Instance(const Platform &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
  };
  jclass clazz;
  jmethodID ctor0Method;

private:
  Platform(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Platform>> cached;

public:
  static Platform &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong nativePeer, jstring name) const;
};
struct Event {
  struct Instance {
    const Event &meta;
    jobject instance;
    Instance(const Event &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
  };
  jclass clazz;
  jmethodID ctor0Method;

private:
  Event(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Event>> cached;

public:
  static Event &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong epochMillis, jlong elapsedNanos, jstring name, jstring data, jobjectArray items) const;
};
struct Layout {
  struct Instance {
    const Layout &meta;
    jobject instance;
    Instance(const Layout &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
  };
  jclass clazz;
  jmethodID ctor0Method;

private:
  Layout(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Layout>> cached;

public:
  static Layout &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jstring name, jlong sizeInBytes, jlong alignment, jobjectArray members) const;
};
struct Member {
  struct Instance {
    const Member &meta;
    jobject instance;
    Instance(const Member &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
  };
  jclass clazz;
  jmethodID ctor0Method;

private:
  Member(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Member>> cached;

public:
  static Member &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jstring name, jlong offsetInBytes, jlong sizeInBytes) const;
};
struct Options {
  struct Instance {
    const Options &meta;
    jobject instance;
    Instance(const Options &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jstring arch(JNIEnv *env) const;
    jbyte target(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID archField;
  jfieldID targetField;
  jmethodID ctor0Method;

private:
  Options(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Options>> cached;

public:
  static Options &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jbyte target, jstring arch) const;
};
struct Compilation {
  struct Instance {
    const Compilation &meta;
    jobject instance;
    Instance(const Compilation &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
  };
  jclass clazz;
  jmethodID ctor0Method;

private:
  Compilation(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Compilation>> cached;

public:
  static Compilation &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jbyteArray program, jobjectArray features, jobjectArray entryArgs, jobjectArray events,
                      jobjectArray layouts, jstring messages) const;
};
struct String {
  struct Instance {
    const String &meta;
    jobject instance;
    Instance(const String &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
  };
  jclass clazz;

private:
  String(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<String>> cached;

public:
  static String &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
};
struct Runnable {
  struct Instance {
    const Runnable &meta;
    jobject instance;
    Instance(const Runnable &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    void run(JNIEnv *env) const;
  };
  jclass clazz;
  jmethodID run_VMethod;

private:
  Runnable(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<Runnable>> cached;

public:
  static Runnable &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
};
struct File {
  struct Instance {
    const File &meta;
    jobject instance;
    Instance(const File &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F &&f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jboolean delete_(JNIEnv *env) const;
  };
  jclass clazz;
  jmethodID delete_ZMethod;

private:
  File(JNIEnv *env, jclass found);
  static thread_local std::vector<std::unique_ptr<File>> cached;

public:
  static File &of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap(JNIEnv *env, jobject instance);
};
} // namespace polyregion::generated