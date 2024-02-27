
#include "polyrt/metal_platform.h"
#include <cstring>

using namespace polyregion::runtime;
using namespace polyregion::runtime::metal;

static constexpr const char *ERROR_PREFIX = "[Metal error] ";

#define NOT_NIL_ERROR(f, error, message) throwIfNil((f), (error), (message), __FILE__, __LINE__)
#define NOT_NIL(f, message) throwIfNil((f), (message), __FILE__, __LINE__)

static std::string to_string(NS::String *str) { return str ? std::string(str->utf8String()) : ""; }

template <typename T> //
static T *throwIfNil(T *t, NS::Error *error, const std::string &message, const char *file, int line) {
  if (!t) {
    if (error) {
      throw std::logic_error(std::string(ERROR_PREFIX) + file + ":" + std::to_string(line) + ": " + message + ": " +
                             to_string(error->localizedDescription()) +
                             "\nSuggestion:" + to_string(error->localizedRecoverySuggestion()));
    } else
      throw std::logic_error(std::string(ERROR_PREFIX) + file + ":" + std::to_string(line) + ": " + message +
                             " and Metal did not provide a reason");
  }
  return t;
}

template <typename T> //
static T *throwIfNil(T *t, const std::string &nown, const char *file, int line) {
  if (!t) {
    throw std::logic_error(std::string(ERROR_PREFIX) + file + ":" + std::to_string(line) + ": " +
                           std::string("Unable to acquire a " + nown));
  }
  return t;
}

MetalPlatform::MetalPlatform() : pool(NS::AutoreleasePool::alloc()->init()) { TRACE(); }
std::string MetalPlatform::name() {
  TRACE();
  return "Metal";
}
std::vector<Property> MetalPlatform::properties() {
  TRACE();
  return {};
}
PlatformKind MetalPlatform::kind() {
  TRACE();
  return PlatformKind::Managed;
}
ModuleFormat MetalPlatform::moduleFormat() {
  TRACE();
  return ModuleFormat::Source;
}
std::vector<std::unique_ptr<Device>> MetalPlatform::enumerate() {
  TRACE();
  std::vector<std::unique_ptr<Device>> devices;
  auto allDevices = MTL::CopyAllDevices();
  for (size_t i = 0; i < allDevices->count(); ++i) {
    auto device = allDevices->object<MTL::Device>(i);
    devices.push_back(std::make_unique<MetalDevice>(device->retain()));
  }
  return devices;
}
MetalPlatform::~MetalPlatform() {
  TRACE();
  pool->release();
}

// ---

MetalDevice::MetalDevice(decltype(device) device_)
    : pool(NS::AutoreleasePool::alloc()->init()), device(device_),
      store(
          ERROR_PREFIX,
          [this](auto &&image) {
            TRACE();
            auto options = MTL::CompileOptions::alloc()->init();
            options->setFastMathEnabled(true);
            NS::Error *error = nil;
            return NOT_NIL_ERROR(
                device->newLibrary(NS::String::string(image.c_str(), NS::StringEncoding::UTF8StringEncoding), //
                                   options,                                                                   //
                                   &error                                                                     //
                                   ),
                error, "Program failed to compile");
          }, //
          [this](auto &&m, auto &&name, auto) {
            auto fn = NOT_NIL(m->newFunction(NS::String::string(name.c_str(), NS::StringEncoding::UTF8StringEncoding)),
                              "function (" + name + ")");
            NS::Error *error = nil;
            return NOT_NIL_ERROR(device->newComputePipelineState(fn, &error), error,
                                 "Function " + name + " failed to resolve");
          }, //
          [&](auto &&m) {
            TRACE();
            m->release();
          }, //
          [&](auto &&f) {
            TRACE();
            f->release();
          }) {
  TRACE();
}

int64_t MetalDevice::id() {
  TRACE();
  return int64_t(device->locationNumber());
}
std::string MetalDevice::name() {
  TRACE();
  return device->name()->utf8String();
}
bool MetalDevice::sharedAddressSpace() {
  TRACE();
  return false;
}
bool MetalDevice::singleEntryPerModule() {
  TRACE();
  return false;
}
std::vector<Property> MetalDevice::properties() {
  TRACE();
  return {{
      {"maxThreadgroupMemoryLength", std::to_string(device->maxThreadgroupMemoryLength())},
      {"maxThreadgroupMemoryLengthWidth", std::to_string(device->maxThreadsPerThreadgroup().width)},
      {"maxThreadgroupMemoryLengthHeight", std::to_string(device->maxThreadsPerThreadgroup().height)},
      {"maxThreadgroupMemoryLengthDepth", std::to_string(device->maxThreadsPerThreadgroup().depth)},
  }};
}
std::vector<std::string> MetalDevice::features() {
  TRACE();
  return {};
}
void MetalDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  store.loadModule(name, image);
}
bool MetalDevice::moduleLoaded(const std::string &name) {
  TRACE();
  return store.moduleLoaded(name);
}
uintptr_t MetalDevice::malloc(size_t size, Access access) {
  TRACE();
  return memoryObjects.malloc(device->newBuffer(size, MTL::ResourceStorageModeShared));
}

void MetalDevice::free(uintptr_t ptr) {
  TRACE();
  if (auto mem = memoryObjects.query(ptr); mem) {
    (*mem)->release();
    memoryObjects.erase(ptr);
  } else
    throw std::logic_error(std::string(ERROR_PREFIX) + "Illegal memory object: " + std::to_string(ptr));
}
std::unique_ptr<DeviceQueue> MetalDevice::createQueue() {
  TRACE();
  return std::make_unique<MetalDeviceQueue>(
      store, NOT_NIL(device->newCommandQueue(), "command queue"), [this](auto &&ptr) {
        if (auto mem = memoryObjects.query(ptr); mem) {
          return *mem;
        } else
          throw std::logic_error(std::string(ERROR_PREFIX) + "Illegal memory object: " + std::to_string(ptr));
      });
}
MetalDevice::~MetalDevice() {
  TRACE();
  device->release();
  pool->release();
}

// ---

MetalDeviceQueue::MetalDeviceQueue(decltype(store) store, decltype(queue) queue,
                                   decltype(queryMemObject) queryMemObject)
    : pool(NS::AutoreleasePool::alloc()->init()), store(store), queue(queue),
      queryMemObject(std::move(queryMemObject)) {
  TRACE();
}
MetalDeviceQueue::~MetalDeviceQueue() {
  TRACE();
  queue->release();
  pool->release();
}
void MetalDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  std::memcpy(queryMemObject(dst)->contents(), src, size);
  if (cb) (*cb)();
}
void MetalDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  std::memcpy(dst, queryMemObject(src)->contents(), size);
  if (cb) (*cb)();
}
void MetalDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                          const std::vector<Type> &types, std::vector<std::byte> argData,
                                          const Policy &policy, const MaybeCallback &cb) {
  TRACE();
  if (types.back() != Type::Void)
    throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported, was " +
                           runtime::typeName(types.back()));
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
    buffer->addCompletedHandler(^void(MTL::CommandBuffer *) {
      (*cb)();
    });
  }
  buffer->commit();
  TRACE();
  buffer->waitUntilCompleted();
}

#undef NOT_NIL_ERROR
#undef NOT_NIL
