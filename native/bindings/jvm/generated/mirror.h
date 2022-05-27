#include <jni.h>
#include <optional>
#include <memory>
namespace polyregion::generated {
struct ByteBuffer {
  struct Instance {
    const ByteBuffer &meta;
    jobject instance;
    Instance(const ByteBuffer &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    
  };
  jclass clazz;
  jmethodID allocate_ILjava_nio_ByteBuffer_Method;
  jmethodID allocateDirect_ILjava_nio_ByteBuffer_Method;
private:
  explicit ByteBuffer(JNIEnv *env);
  static std::unique_ptr<ByteBuffer> cached;
public:
  static ByteBuffer& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  jobject allocate(JNIEnv *env, jint arg0) const;
  jobject allocateDirect(JNIEnv *env, jint arg0) const;
};
struct Property {
  struct Instance {
    const Property &meta;
    jobject instance;
    Instance(const Property &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jstring key(JNIEnv *env) const;
    jstring value(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID keyField;
  jfieldID valueField;
  jmethodID ctor0Method;
private:
  explicit Property(JNIEnv *env);
  static std::unique_ptr<Property> cached;
public:
  static Property& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jstring key, jstring value) const;
};
struct Dim3 {
  struct Instance {
    const Dim3 &meta;
    jobject instance;
    Instance(const Dim3 &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
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
  explicit Dim3(JNIEnv *env);
  static std::unique_ptr<Dim3> cached;
public:
  static Dim3& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong x, jlong y, jlong z) const;
};
struct Policy {
  struct Instance {
    const Policy &meta;
    jobject instance;
    Instance(const Policy &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    Dim3::Instance global(JNIEnv *env, const Dim3& clazz) const;
    Dim3::Instance local(JNIEnv *env, const Dim3& clazz) const;
    jobject local(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID globalField;
  jfieldID localField;
  jmethodID ctor0Method;
  jmethodID local_Ljava_util_Optional_Method;
private:
  explicit Policy(JNIEnv *env);
  static std::unique_ptr<Policy> cached;
public:
  static Policy& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jobject global) const;
};
struct Queue {
  struct Instance {
    const Queue &meta;
    jobject instance;
    Instance(const Queue &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jlong nativePeer(JNIEnv *env) const;
    void close(JNIEnv *env) const;
    void enqueueHostToDeviceAsync(JNIEnv *env, jobject src, jlong dst, jint size, jobject cb) const;
    void enqueueDeviceToHostAsync(JNIEnv *env, jlong src, jobject dst, jint size, jobject cb) const;
    void enqueueInvokeAsync(JNIEnv *env, jstring moduleName, jstring symbol, jobject args, jobject rtn, jobject policy, jobject cb) const;
    void enqueueInvokeAsync(JNIEnv *env, jstring moduleName, jstring symbol, jbyteArray argTys, jobjectArray argBuffers, jbyte rtnTy, jobject rtnBuffer, jobject policy, jobject cb) const;
  };
  jclass clazz;
  jfieldID nativePeerField;
  jmethodID ctor0Method;
  jmethodID close_VMethod;
  jmethodID enqueueHostToDeviceAsync_Ljava_nio_ByteBuffer_JILjava_lang_Runnable_VMethod;
  jmethodID enqueueDeviceToHostAsync_JLjava_nio_ByteBuffer_ILjava_lang_Runnable_VMethod;
  jmethodID enqueueInvokeAsync_Ljava_lang_String_Ljava_lang_String_Ljava_util_List_Lpolyregion_jvm_runtime_Arg_Lpolyregion_jvm_runtime_Policy_Ljava_lang_Runnable_VMethod;
  jmethodID enqueueInvokeAsync_Ljava_lang_String_Ljava_lang_String_aBaLjava_nio_ByteBuffer_BLjava_nio_ByteBuffer_Lpolyregion_jvm_runtime_Policy_Ljava_lang_Runnable_VMethod;
private:
  explicit Queue(JNIEnv *env);
  static std::unique_ptr<Queue> cached;
public:
  static Queue& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong nativePeer) const;
};
struct Device {
  struct Instance {
    const Device &meta;
    jobject instance;
    Instance(const Device &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jlong nativePeer(JNIEnv *env) const;
    jlong id(JNIEnv *env) const;
    jstring name(JNIEnv *env) const;
    void loadModule(JNIEnv *env, jstring name, jbyteArray image) const;
    jobjectArray properties(JNIEnv *env) const;
    void close(JNIEnv *env) const;
    void free(JNIEnv *env, jlong data) const;
    Queue::Instance createQueue(JNIEnv *env, const Queue& clazz_) const;
    jlong malloc(JNIEnv *env, jlong size, jobject access) const;
  };
  jclass clazz;
  jfieldID nativePeerField;
  jfieldID idField;
  jfieldID nameField;
  jmethodID ctor0Method;
  jmethodID loadModule_Ljava_lang_String_aBVMethod;
  jmethodID properties_aLpolyregion_jvm_runtime_Property_Method;
  jmethodID close_VMethod;
  jmethodID free_JVMethod;
  jmethodID createQueue_Lpolyregion_jvm_runtime_Device_Queue_Method;
  jmethodID malloc_JLpolyregion_jvm_runtime_Access_JMethod;
private:
  explicit Device(JNIEnv *env);
  static std::unique_ptr<Device> cached;
public:
  static Device& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong nativePeer, jlong id, jstring name) const;
};
struct Runtime {
  struct Instance {
    const Runtime &meta;
    jobject instance;
    Instance(const Runtime &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jlong nativePeer(JNIEnv *env) const;
    jstring name(JNIEnv *env) const;
    jobjectArray properties(JNIEnv *env) const;
    void close(JNIEnv *env) const;
    jobjectArray devices(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID nativePeerField;
  jfieldID nameField;
  jmethodID ctor0Method;
  jmethodID properties_aLpolyregion_jvm_runtime_Property_Method;
  jmethodID close_VMethod;
  jmethodID devices_aLpolyregion_jvm_runtime_Device_Method;
private:
  explicit Runtime(JNIEnv *env);
  static std::unique_ptr<Runtime> cached;
public:
  static Runtime& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong nativePeer, jstring name) const;
};
struct Event {
  struct Instance {
    const Event &meta;
    jobject instance;
    Instance(const Event &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jlong epochMillis(JNIEnv *env) const;
    jlong elapsedNanos(JNIEnv *env) const;
    jstring name(JNIEnv *env) const;
    jstring data(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID epochMillisField;
  jfieldID elapsedNanosField;
  jfieldID nameField;
  jfieldID dataField;
  jmethodID ctor0Method;
private:
  explicit Event(JNIEnv *env);
  static std::unique_ptr<Event> cached;
public:
  static Event& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong epochMillis, jlong elapsedNanos, jstring name, jstring data) const;
};
struct Layout {
  struct Instance {
    const Layout &meta;
    jobject instance;
    Instance(const Layout &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jobjectArray name(JNIEnv *env) const;
    jlong sizeInBytes(JNIEnv *env) const;
    jlong alignment(JNIEnv *env) const;
    jobjectArray members(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID nameField;
  jfieldID sizeInBytesField;
  jfieldID alignmentField;
  jfieldID membersField;
  jmethodID ctor0Method;
private:
  explicit Layout(JNIEnv *env);
  static std::unique_ptr<Layout> cached;
public:
  static Layout& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jobjectArray name, jlong sizeInBytes, jlong alignment, jobjectArray members) const;
};
struct Member {
  struct Instance {
    const Member &meta;
    jobject instance;
    Instance(const Member &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jstring name(JNIEnv *env) const;
    jlong offsetInBytes(JNIEnv *env) const;
    jlong sizeInBytes(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID nameField;
  jfieldID offsetInBytesField;
  jfieldID sizeInBytesField;
  jmethodID ctor0Method;
private:
  explicit Member(JNIEnv *env);
  static std::unique_ptr<Member> cached;
public:
  static Member& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jstring name, jlong offsetInBytes, jlong sizeInBytes) const;
};
struct Options {
  struct Instance {
    const Options &meta;
    jobject instance;
    Instance(const Options &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jbyte target(JNIEnv *env) const;
    jstring arch(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID targetField;
  jfieldID archField;
  jmethodID ctor0Method;
private:
  explicit Options(JNIEnv *env);
  static std::unique_ptr<Options> cached;
public:
  static Options& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jbyte target, jstring arch) const;
};
struct Compilation {
  struct Instance {
    const Compilation &meta;
    jobject instance;
    Instance(const Compilation &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jbyteArray program(JNIEnv *env) const;
    jobjectArray events(JNIEnv *env) const;
    jobjectArray layouts(JNIEnv *env) const;
    jstring messages(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID programField;
  jfieldID eventsField;
  jfieldID layoutsField;
  jfieldID messagesField;
  jmethodID ctor0Method;
private:
  explicit Compilation(JNIEnv *env);
  static std::unique_ptr<Compilation> cached;
public:
  static Compilation& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jbyteArray program, jobjectArray events, jobjectArray layouts, jstring messages) const;
};
struct String {
  struct Instance {
    const String &meta;
    jobject instance;
    Instance(const String &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    
  };
  jclass clazz;
private:
  explicit String(JNIEnv *env);
  static std::unique_ptr<String> cached;
public:
  static String& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  
};
struct Runnable {
  struct Instance {
    const Runnable &meta;
    jobject instance;
    Instance(const Runnable &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    void run(JNIEnv *env) const;
  };
  jclass clazz;
  jmethodID run_VMethod;
private:
  explicit Runnable(JNIEnv *env);
  static std::unique_ptr<Runnable> cached;
public:
  static Runnable& of(JNIEnv *env);
  static void drop(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  
};
}// polyregion::generated