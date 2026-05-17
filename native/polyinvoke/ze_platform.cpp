#include "polyinvoke/ze_platform.h"

#include <cstring>

#include "magic_enum/magic_enum.hpp"

#include "dl_util.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::ze;

static constexpr auto PREFIX = "LevelZero";

#define CHECKED(f__)                                                                                                                       \
  do {                                                                                                                                     \
    ze_result_t result__ = (f__);                                                                                                          \
    if (result__ != ZE_RESULT_SUCCESS) {                                                                                                   \
      POLYINVOKE_FATAL(PREFIX, "%s:%d: ze_result=0x%08X; callsite: `%s`", __FILE__, __LINE__, result__, #f__);                             \
    }                                                                                                                                      \
  } while (0)

std::variant<std::string, std::unique_ptr<Platform>> ZePlatform::create() {
#ifdef _WIN32
  void *lib = dl::open_first({"ze_loader.dll"});
#elif defined(__APPLE__)
  void *lib = dl::open_first({"libze_loader.dylib"});
#else
  void *lib = dl::open_first({"libze_loader.so.1", "libze_loader.so"});
#endif
  if (!lib) return "Level Zero: failed to open libze_loader dynamic library, no Level Zero driver present?";
  zeew_ze_resolve(dl::lookup, lib);
  if (const auto result = zeInit(0); result != ZE_RESULT_SUCCESS) {
    return "Level Zero: zeInit failed with code " + std::to_string(result);
  }
  uint32_t driverCount = 0;
  if (const auto r = zeDriverGet(&driverCount, nullptr); r != ZE_RESULT_SUCCESS || driverCount == 0) {
    return "Level Zero: zeDriverGet returned no drivers";
  }
  std::vector<ze_driver_handle_t> drivers(driverCount);
  CHECKED(zeDriverGet(&driverCount, drivers.data()));
  // XXX First driver wins
  return std::unique_ptr<Platform>(new ZePlatform(drivers.front()));
}

ZePlatform::ZePlatform(ze_driver_handle_t driver) : driver(driver) { POLYINVOKE_TRACE(); }
std::string ZePlatform::name() {
  POLYINVOKE_TRACE();
  return "Level Zero";
}
std::vector<Property> ZePlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}
PlatformKind ZePlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::Managed;
}
std::vector<std::unique_ptr<Device>> ZePlatform::enumerate() {
  POLYINVOKE_TRACE();
  uint32_t count = 0;
  CHECKED(zeDeviceGet(driver, &count, nullptr));
  if (count == 0) return {};
  std::vector<ze_device_handle_t> handles(count);
  CHECKED(zeDeviceGet(driver, &count, handles.data()));
  std::vector<std::unique_ptr<Device>> devices;
  devices.reserve(count);
  for (auto h : handles)
    devices.push_back(std::make_unique<ZeDevice>(driver, h));
  return devices;
}

// ---

namespace {

ze_context_handle_t createContext(ze_driver_handle_t driver) {
  ze_context_desc_t desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx = {};
  CHECKED(zeContextCreate(driver, &desc, &ctx));
  return ctx;
}

std::string queryDeviceName(ze_device_handle_t device) {
  ze_device_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  if (zeDeviceGetProperties(device, &props) != ZE_RESULT_SUCCESS) return "(unknown)";
  // `name` is a fixed-size char array; bound the read with strnlen so a missing terminator
  // can't run off the end.
  return std::string(props.name, ::strnlen(props.name, sizeof(props.name)));
}

} // namespace

ZeDevice::ZeDevice(ze_driver_handle_t driver, ze_device_handle_t device)
    : driver(driver), device(device), context([this]() { return createContext(this->driver); },
                                              [](auto ctx) {
                                                POLYINVOKE_TRACE();
                                                CHECKED(zeContextDestroy(ctx));
                                              }),
      store(
          PREFIX,
          [this](auto &&image) {
            POLYINVOKE_TRACE();
            context.touch();
            ze_module_desc_t desc{ZE_STRUCTURE_TYPE_MODULE_DESC,
                                  nullptr,
                                  ZE_MODULE_FORMAT_IL_SPIRV,
                                  image.size(),
                                  reinterpret_cast<const uint8_t *>(image.data()),
                                  nullptr,
                                  nullptr};
            ze_module_handle_t module = {};
            ze_module_build_log_handle_t buildLog = {};
            if (const auto r = zeModuleCreate(*context, this->device, &desc, &module, &buildLog); r != ZE_RESULT_SUCCESS) {
              size_t logSize = 0;
              std::string log;
              if (buildLog && zeModuleBuildLogGetString(buildLog, &logSize, nullptr) == ZE_RESULT_SUCCESS && logSize > 0) {
                log.resize(logSize);
                zeModuleBuildLogGetString(buildLog, &logSize, log.data());
                // zeModuleBuildLogGetString writes the trailing NUL inside logSize.
                if (!log.empty() && log.back() == '\0') log.pop_back();
              }
              if (buildLog) zeModuleBuildLogDestroy(buildLog);
              POLYINVOKE_FATAL(PREFIX, "zeModuleCreate failed (0x%08X): %s", r, log.c_str());
            }
            if (buildLog) zeModuleBuildLogDestroy(buildLog);
            return module;
          },
          [](auto &&module, auto &&name, auto &&) {
            POLYINVOKE_TRACE();
            ze_kernel_desc_t desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name.c_str()};
            ze_kernel_handle_t kernel = {};
            CHECKED(zeKernelCreate(module, &desc, &kernel));
            return kernel;
          },
          [&](auto &&module) {
            POLYINVOKE_TRACE();
            CHECKED(zeModuleDestroy(module));
          },
          [&](auto &&kernel) {
            POLYINVOKE_TRACE();
            CHECKED(zeKernelDestroy(kernel));
          }) {
  POLYINVOKE_TRACE();
  deviceName = queryDeviceName(device);
}

ZeDevice::~ZeDevice() { POLYINVOKE_TRACE(); }

int64_t ZeDevice::id() {
  POLYINVOKE_TRACE();
  return reinterpret_cast<intptr_t>(device);
}
std::string ZeDevice::name() {
  POLYINVOKE_TRACE();
  return deviceName;
}
ModuleFormat ZeDevice::moduleFormat() {
  POLYINVOKE_TRACE();
  return ModuleFormat::SPIRV_Kernel;
}
bool ZeDevice::sharedAddressSpace() {
  POLYINVOKE_TRACE();
  return false;
}
bool ZeDevice::singleEntryPerModule() {
  POLYINVOKE_TRACE();
  return false;
}
std::vector<Property> ZeDevice::properties() {
  POLYINVOKE_TRACE();
  return {};
}
std::vector<std::string> ZeDevice::features() {
  POLYINVOKE_TRACE();
  std::vector<std::string> out{"levelzero", "spirv_kernel"};
  ze_device_module_properties_t mod{};
  mod.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
  if (zeDeviceGetModuleProperties(device, &mod) == ZE_RESULT_SUCCESS) {
    if (mod.flags & ZE_DEVICE_MODULE_FLAG_FP16) out.emplace_back("fp16");
    if (mod.flags & ZE_DEVICE_MODULE_FLAG_FP64) out.emplace_back("fp64");
    if (mod.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) out.emplace_back("int64");
  }
  return out;
}
void ZeDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  store.loadModule(name, image);
}
bool ZeDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  return store.moduleLoaded(name);
}
uintptr_t ZeDevice::mallocDevice(size_t size, Access) {
  POLYINVOKE_TRACE();
  context.touch();
  if (size == 0) POLYINVOKE_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void *ptr = nullptr;
  CHECKED(zeMemAllocDevice(*context, &desc, size, /*alignment*/ 0, device, &ptr));
  return reinterpret_cast<uintptr_t>(ptr);
}
void ZeDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  CHECKED(zeMemFree(*context, reinterpret_cast<void *>(ptr)));
}
std::optional<void *> ZeDevice::mallocShared(size_t size, Access) {
  POLYINVOKE_TRACE();
  context.touch();
  if (size == 0) POLYINVOKE_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  ze_device_mem_alloc_desc_t deviceDesc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ze_host_mem_alloc_desc_t hostDesc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  void *ptr = nullptr;
  CHECKED(zeMemAllocShared(*context, &deviceDesc, &hostDesc, size, /*alignment*/ 0, device, &ptr));
  return ptr;
}
void ZeDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  CHECKED(zeMemFree(*context, ptr));
}
std::unique_ptr<DeviceQueue> ZeDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYINVOKE_TRACE();
  context.touch();
  return std::make_unique<ZeDeviceQueue>(timeout, store, *context, device);
}

// ---

ZeDeviceQueue::ZeDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store, ze_context_handle_t context,
                             ze_device_handle_t device)
    : latch(timeout), store(store) {
  POLYINVOKE_TRACE();
  // Async immediate command list: appends submit straight to the device without an explicit
  // queue.  Ordinal 0 is always the default compute group.
  ze_command_queue_desc_t desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                               nullptr,
                               /*ordinal=*/0,
                               /*index=*/0,
                               /*flags=*/0,
                               ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                               ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECKED(zeCommandListCreateImmediate(context, device, &desc, &cmdList));

  // Single-slot host-visible event pool.  Every append uses this event as its signal slot;
  // waitAndFire host-syncs and resets it before the next op.  v1.0-compatible (no
  // zeCommandListHostSynchronize).
  ze_event_pool_desc_t poolDesc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, /*count=*/1};
  CHECKED(zeEventPoolCreate(context, &poolDesc, /*numDevices=*/1, &device, &eventPool));
  ze_event_desc_t eventDesc{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, /*index=*/0,
                            /*signal=*/0, ZE_EVENT_SCOPE_FLAG_HOST};
  CHECKED(zeEventCreate(eventPool, &eventDesc, &event));
}
ZeDeviceQueue::~ZeDeviceQueue() {
  POLYINVOKE_TRACE();
  // Every enqueue host-syncs inline before returning, so the cmd list has no pending work
  // here; safe to destroy directly.
  if (event) CHECKED(zeEventDestroy(event));
  if (eventPool) CHECKED(zeEventPoolDestroy(eventPool));
  CHECKED(zeCommandListDestroy(cmdList));
}
void ZeDeviceQueue::waitAndFire(const MaybeCallback &cb) {
  CHECKED(zeEventHostSynchronize(event, UINT64_MAX));
  CHECKED(zeEventHostReset(event));
  if (cb) (*cb)();
}
void ZeDeviceQueue::appendCopyAndWait(void *dst, const void *src, size_t size, const MaybeCallback &cb) {
  CHECKED(zeCommandListAppendMemoryCopy(cmdList, dst, src, size, event, 0, nullptr));
  waitAndFire(cb);
}
void ZeDeviceQueue::enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                               const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  appendCopyAndWait(reinterpret_cast<void *>(dst + dstOffset), reinterpret_cast<void *>(src + srcOffset), size, cb);
}
void ZeDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  appendCopyAndWait(reinterpret_cast<void *>(dst + dstOffset), src, size, cb);
}
void ZeDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  appendCopyAndWait(dst, reinterpret_cast<void *>(src + srcOffset), size, cb);
}
void ZeDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                       std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  if (types.back() != Type::Void)
    POLYINVOKE_FATAL(PREFIX, "Non-void return type not supported: %s", magic_enum::enum_name(types.back()).data());
  auto kernel = store.resolveFunction(moduleName, symbol, types);

  auto grid = policy.global;
  auto [block, sharedMem] = policy.local.value_or(std::pair{Dim3{}, size_t{0}});
  CHECKED(zeKernelSetGroupSize(kernel, static_cast<uint32_t>(block.x), static_cast<uint32_t>(block.y), static_cast<uint32_t>(block.z)));

  auto argPtrs = detail::argDataAsPointers(types, argData);
  uint32_t argIdx = 0;
  for (size_t i = 0; i < types.size(); ++i) {
    if (types[i] == Type::Void) continue;
    if (types[i] == Type::Scratch) {
      // Workgroup scratch: pass size only, kernel slot consumes a void* argument shape.
      CHECKED(zeKernelSetArgumentValue(kernel, argIdx, sharedMem, nullptr));
    } else {
      CHECKED(zeKernelSetArgumentValue(kernel, argIdx, byteOfType(types[i]), argPtrs[i]));
    }
    ++argIdx;
  }

  ze_group_count_t groups{static_cast<uint32_t>(grid.x), static_cast<uint32_t>(grid.y), static_cast<uint32_t>(grid.z)};
  CHECKED(zeCommandListAppendLaunchKernel(cmdList, kernel, &groups, event, 0, nullptr));
  waitAndFire(cb);
}
void ZeDeviceQueue::enqueueWaitBlocking() {
  POLYINVOKE_TRACE();
  // No-op: every enqueueXxxAsync host-syncs inline before returning, so the queue is always
  // drained when this is reached.  Keeps the API contract for callers that prefer explicit
  // wait semantics.
}

#undef CHECKED
