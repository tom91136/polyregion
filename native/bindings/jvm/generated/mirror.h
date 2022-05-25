#include <jni.h>
#include <optional>
namespace polyregion::generated {
struct ByteBuffer {
  struct Instance {
    const ByteBuffer &meta;
    jobject instance;
    Instance(const ByteBuffer &meta, jobject instance);
    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
    jbyteArray hb(JNIEnv *env) const;
    jint offset(JNIEnv *env) const;
    jboolean isReadOnly(JNIEnv *env) const;
    jboolean bigEndian(JNIEnv *env) const;
    jboolean nativeByteOrder(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID hbField;
  jfieldID offsetField;
  jfieldID isReadOnlyField;
  jfieldID bigEndianField;
  jfieldID nativeByteOrderField;
  jmethodID ctor0Method;
  jmethodID ctor1Method;
  jmethodID ctor2Method;
  jmethodID allocate_ILjava_nio_ByteBuffer_Method;
  jmethodID allocateDirect_ILjava_nio_ByteBuffer_Method;
  explicit ByteBuffer(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  jobject allocate(JNIEnv *env, jint arg0) const;
  jobject allocateDirect(JNIEnv *env, jint arg0) const;
  Instance operator()(JNIEnv *env, jbyteArray arg0, jlong arg1, jint arg2, jobject arg3) const;
  Instance operator()(JNIEnv *env, jint arg0, jint arg1, jint arg2, jint arg3, jobject arg4) const;
  Instance operator()(JNIEnv *env, jint arg0, jint arg1, jint arg2, jint arg3, jbyteArray arg4, jint arg5, jobject arg6) const;
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
  explicit Property(JNIEnv *env);
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
  explicit Dim3(JNIEnv *env);
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
  explicit Policy(JNIEnv *env);
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
    void enqueueHostToDeviceAsync(JNIEnv *env, jobject src, jlong dst, jint size, jobject cb) const;
    void enqueueInvokeAsync(JNIEnv *env, jstring moduleName, jstring symbol, jbyteArray argTys, jobjectArray argBuffers, jbyte rtnTy, jobject rtnBuffer, jobject policy, jobject cb) const;
    void enqueueInvokeAsync(JNIEnv *env, jstring moduleName, jstring symbol, jobject args, jobject rtn, jobject policy, jobject cb) const;
    void enqueueDeviceToHostAsync(JNIEnv *env, jlong src, jobject dst, jint size, jobject cb) const;
  };
  jclass clazz;
  jfieldID nativePeerField;
  jmethodID ctor0Method;
  jmethodID enqueueHostToDeviceAsync_Ljava_nio_ByteBuffer_JILjava_lang_Runnable_VMethod;
  jmethodID enqueueInvokeAsync_Ljava_lang_String_Ljava_lang_String_aBaLjava_nio_ByteBuffer_BLjava_nio_ByteBuffer_Lpolyregion_jvm_runtime_Policy_Ljava_lang_Runnable_VMethod;
  jmethodID enqueueInvokeAsync_Ljava_lang_String_Ljava_lang_String_Ljava_util_List_Lpolyregion_jvm_runtime_Arg_Lpolyregion_jvm_runtime_Policy_Ljava_lang_Runnable_VMethod;
  jmethodID enqueueDeviceToHostAsync_JLjava_nio_ByteBuffer_ILjava_lang_Runnable_VMethod;
  explicit Queue(JNIEnv *env);
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
    jobjectArray properties(JNIEnv *env) const;
    jlong malloc(JNIEnv *env, jlong size, jobject access) const;
    Queue::Instance createQueue(JNIEnv *env, const Queue& clazz_) const;
    void loadModule(JNIEnv *env, jstring name, jbyteArray image) const;
    void free(JNIEnv *env, jlong data) const;
  };
  jclass clazz;
  jfieldID nativePeerField;
  jfieldID idField;
  jfieldID nameField;
  jfieldID propertiesField;
  jmethodID ctor0Method;
  jmethodID malloc_JLpolyregion_jvm_runtime_Access_JMethod;
  jmethodID createQueue_Lpolyregion_jvm_runtime_Device_Queue_Method;
  jmethodID loadModule_Ljava_lang_String_aBVMethod;
  jmethodID free_JVMethod;
  explicit Device(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong nativePeer, jlong id, jstring name, jobjectArray properties) const;
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
    jobjectArray devices(JNIEnv *env) const;
  };
  jclass clazz;
  jfieldID nativePeerField;
  jfieldID nameField;
  jfieldID propertiesField;
  jmethodID ctor0Method;
  jmethodID devices_aLpolyregion_jvm_runtime_Device_Method;
  explicit Runtime(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jlong nativePeer, jstring name, jobjectArray properties) const;
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
  explicit Event(JNIEnv *env);
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
  explicit Layout(JNIEnv *env);
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
  explicit Member(JNIEnv *env);
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
  explicit Options(JNIEnv *env);
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
  explicit Compilation(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  Instance operator()(JNIEnv *env, jbyteArray program, jobjectArray events, jobjectArray layouts, jstring messages) const;
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
  explicit Runnable(JNIEnv *env);
  Instance wrap (JNIEnv *env, jobject instance);
  
};
}// polyregion::generated