#include <cstring>

#include "polyinvoke/metal_platform.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::metal;

static constexpr const char *PREFIX = "Metal";

#define NOT_NIL_ERROR(f, error, message) throwIfNil((f), (error), (message), __FILE__, __LINE__)
#define NOT_NIL(f, message) throwIfNil((f), (message), __FILE__, __LINE__)

static std::string to_string(NS::String *str) { return str ? std::string(str->utf8String()) : ""; }

template <typename T> //
static T *throwIfNil(T *t, NS::Error *error, const std::string &message, const char *file, int line) {
  if (!t) {
    if (error) {
      POLYINVOKE_FATAL(PREFIX, "%s:%d: %s: %s\nSuggestion:%s\n", file, line, message.c_str(), error->localizedDescription()->utf8String(),
                   error->localizedRecoverySuggestion()->utf8String());
    } else POLYINVOKE_FATAL(PREFIX, "%s:%d: %s and Metal did not provide a reason\n", file, line, message.c_str());
  }
  return t;
}

template <typename T> //
static T *throwIfNil(T *t, const std::string &nown, const char *file, int line) {
  if (!t) {
    POLYINVOKE_FATAL(PREFIX, "%s:%d: Unable to acquire a %s", file, line, nown.c_str());
  }
  return t;
}

std::variant<std::string, std::unique_ptr<Platform>> MetalPlatform::create() { return std::unique_ptr<Platform>(new MetalPlatform()); }
MetalPlatform::MetalPlatform() : pool(NS::AutoreleasePool::alloc()->init()) { POLYINVOKE_TRACE(); }
std::string MetalPlatform::name() {
  POLYINVOKE_TRACE();
  return "Metal";
}
std::vector<Property> MetalPlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}
PlatformKind MetalPlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::Managed;
}
ModuleFormat MetalPlatform::moduleFormat() {
  POLYINVOKE_TRACE();
  return ModuleFormat::Source;
}
std::vector<std::unique_ptr<Device>> MetalPlatform::enumerate() {
  POLYINVOKE_TRACE();
  std::vector<std::unique_ptr<Device>> devices;
  auto allDevices = MTL::CopyAllDevices();
  for (size_t i = 0; i < allDevices->count(); ++i) {
    auto device = allDevices->object<MTL::Device>(i);
    devices.push_back(std::make_unique<MetalDevice>(device->retain()));
  }
  return devices;
}
MetalPlatform::~MetalPlatform() {
  POLYINVOKE_TRACE();
  pool->release();
}

// ---

MetalDevice::MetalDevice(decltype(device) device_)
    : pool(NS::AutoreleasePool::alloc()->init()), device(device_),
      store(
          PREFIX,
          [this](auto &&image) {
            POLYINVOKE_TRACE();
            auto options = MTL::CompileOptions::alloc()->init();
            options->setFastMathEnabled(true);
            NS::Error *error = nil;
            return NOT_NIL_ERROR(device->newLibrary(NS::String::string(image.c_str(), NS::StringEncoding::UTF8StringEncoding), //
                                                    options,                                                                   //
                                                    &error                                                                     //
                                                    ),
                                 error, "Program failed to compile");
          }, //
          [this](auto &&m, auto &&name, auto) {
            auto fn = NOT_NIL(m->newFunction(NS::String::string(name.c_str(), NS::StringEncoding::UTF8StringEncoding)),
                              "function (" + name + ")");
            NS::Error *error = nil;
            return NOT_NIL_ERROR(device->newComputePipelineState(fn, &error), error, "Function " + name + " failed to resolve");
          }, //
          [&](auto &&m) {
            POLYINVOKE_TRACE();
            m->release();
          }, //
          [&](auto &&f) {
            POLYINVOKE_TRACE();
            f->release();
          }) {
  POLYINVOKE_TRACE();
}

int64_t MetalDevice::id() {
  POLYINVOKE_TRACE();
  return int64_t(device->locationNumber());
}
std::string MetalDevice::name() {
  POLYINVOKE_TRACE();
  return device->name()->utf8String();
}
bool MetalDevice::sharedAddressSpace() {
  POLYINVOKE_TRACE();
  return false;
}
bool MetalDevice::singleEntryPerModule() {
  POLYINVOKE_TRACE();
  return false;
}
std::vector<Property> MetalDevice::properties() {
  POLYINVOKE_TRACE();
  return {{
      {"maxThreadgroupMemoryLength", std::to_string(device->maxThreadgroupMemoryLength())},
      {"maxThreadgroupMemoryLengthWidth", std::to_string(device->maxThreadsPerThreadgroup().width)},
      {"maxThreadgroupMemoryLengthHeight", std::to_string(device->maxThreadsPerThreadgroup().height)},
      {"maxThreadgroupMemoryLengthDepth", std::to_string(device->maxThreadsPerThreadgroup().depth)},
  }};
}
std::vector<std::string> MetalDevice::features() {
  POLYINVOKE_TRACE();
  return {};
}
void MetalDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  store.loadModule(name, image);
}
bool MetalDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  return store.moduleLoaded(name);
}
uintptr_t MetalDevice::mallocDevice(size_t size, Access access) {
  POLYINVOKE_TRACE();
  return memoryObjects.malloc(device->newBuffer(size, MTL::ResourceStorageModeShared));
}
void MetalDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  if (auto mem = memoryObjects.query(ptr); mem) {
    (*mem)->release();
    memoryObjects.erase(ptr);
  } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %lu", ptr);
}
std::optional<void *> MetalDevice::mallocShared(size_t size, Access access) {
  POLYINVOKE_TRACE();
  return std::nullopt;
}
void MetalDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  POLYINVOKE_FATAL(PREFIX, "Unsupported operation on %p", ptr);
}
std::unique_ptr<DeviceQueue> MetalDevice::createQueue(const std::chrono::duration<int64_t> &) {
  POLYINVOKE_TRACE();
  return std::make_unique<MetalDeviceQueue>(store, NOT_NIL(device->newCommandQueue(), "command queue"), [this](auto &&ptr) {
    if (auto mem = memoryObjects.query(ptr); mem) {
      return *mem;
    } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %lu", ptr);
  });
}
MetalDevice::~MetalDevice() {
  POLYINVOKE_TRACE();
  device->release();
  pool->release();
}

// ---

MetalDeviceQueue::MetalDeviceQueue(decltype(store) store, decltype(queue) queue, decltype(queryMemObject) queryMemObject)
    : pool(NS::AutoreleasePool::alloc()->init()), store(store), queue(queue), queryMemObject(std::move(queryMemObject)) {
  POLYINVOKE_TRACE();
}
MetalDeviceQueue::~MetalDeviceQueue() {
  POLYINVOKE_TRACE();
  queue->release();
  pool->release();
}
void MetalDeviceQueue::enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                 const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  std::memcpy(static_cast<char *>(queryMemObject(dst)->contents()) + dstOffset, //
              static_cast<char *>(queryMemObject(src)->contents()) + srcOffset, size);
  if (cb) (*cb)();
}
void MetalDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  std::memcpy(static_cast<char *>(queryMemObject(dst)->contents()) + dstOffset, src, size);
  if (cb) (*cb)();
}
void MetalDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  std::memcpy(dst, static_cast<char *>(queryMemObject(src)->contents()) + srcOffset, size);
  if (cb) (*cb)();
}
void MetalDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                          std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  if (types.back() != Type::Void) POLYINVOKE_FATAL(PREFIX, "Non-void return type not supported: %s", to_string(types.back()).data());
  auto kernel = store.resolveFunction(moduleName, symbol, types);

  auto args = detail::argDataAsPointers(types, argData);
  auto [local, sharedMem] = policy.local.value_or(std::pair{Dim3{}, 0});

  auto buffer = queue->commandBuffer();
  auto encoder = buffer->computeCommandEncoder(MTL::DispatchTypeSerial);
  encoder->setComputePipelineState(kernel);
  // last arg is the return, void assertion should have been done before this
  for (NS::UInteger i = 0; i < types.size() - 1; ++i) {
    auto rawPtr = args[i];
    auto tpe = types[i];
    switch (tpe) {
      case Type::Ptr: {
        static_assert(byteOfType(Type::Ptr) == sizeof(uintptr_t));
        uintptr_t ptr = {};
        std::memcpy(&ptr, rawPtr, byteOfType(Type::Ptr));
        encoder->setBuffer(queryMemObject(ptr), 0, i);
      } break;
      case Type::Scratch: {
        encoder->setThreadgroupMemoryLength(sharedMem, i);
        break;
      }
      default: {
        encoder->setBytes(rawPtr, byteOfType(tpe), i);
        break;
      }
    }
  }

  encoder->dispatchThreadgroups(MTL::Size::Make(policy.global.x, policy.global.y, policy.global.z),
                                MTL::Size::Make(local.x, local.y, local.z));
  encoder->endEncoding();
  if (cb) {
    buffer->addCompletedHandler([&](MTL::CommandBuffer *) -> void { (*cb)(); });
  }
  buffer->commit();
  POLYINVOKE_TRACE();
  buffer->waitUntilCompleted();
}

void MetalDeviceQueue::enqueueWaitBlocking() {
  POLYINVOKE_TRACE();
  queue->commandBuffer()->waitUntilCompleted();
}

#undef NOT_NIL_ERROR
#undef NOT_NIL
