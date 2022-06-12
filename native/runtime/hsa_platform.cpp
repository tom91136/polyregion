#include "hsa_platform.h"
#include "utils.hpp"
#include <cstring>

using namespace polyregion::runtime;
using namespace polyregion::runtime::hsa;

#define CHECKED(m, f) checked((f), m, __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[HSA error] ";

static void checked(hsa_status_t result, const char *msg, const char *file, int line) {
  if (result != HSA_STATUS_SUCCESS) {
    const char *string = nullptr;
    hsa_status_string(result, &string);
    throw std::logic_error(std::string(ERROR_PREFIX) + file + ":" + std::to_string(line) + ": " +
                           (string ? std::string(string) : "(hsa_status_string returned NULL)"));
  }
}

HsaPlatform::HsaPlatform() {
  TRACE();
  hsaew_open("/opt/rocm/hsa/lib/libhsa-runtime64.so");
  CHECKED("HSA_initialisation", hsa_init());

  hsa_amd_register_system_event_handler(
      [](const hsa_amd_event_t *event, void *data) -> hsa_status_t {
        if (event->event_type == HSA_AMD_GPU_MEMORY_FAULT_EVENT) {
          std::string message = std::string(ERROR_PREFIX) + "Memory fault at 0x" +
                                hex((uintptr_t)event->memory_fault.virtual_address) + ". Reason:";
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT) //
            message += "Page not present or supervisor privilege. ";                         //
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_READ_ONLY)        //
            message += "Write access to a read-only page. ";                                 //
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_NX)               //
            message += "Execute access to a page marked NX. ";                               //
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_HOST_ONLY)        //
            message += "Host access only. ";                                                 //
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_DRAMECC)          //
            message += "ECC failure (if supported by HW). ";                                 //
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_IMPRECISE)        //
            message += "Can't determine the exact fault address. ";                          //
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_SRAMECC)          //
            message += "SRAM ECC failure (ie registers, no fault address). ";                //
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_HANG)             //
            message += "GPU reset following unspecified hang. ";                             //
          fprintf(stderr, "%s\n", message.c_str());
          return HSA_STATUS_ERROR;
        }
        return HSA_STATUS_SUCCESS;
      },
      nullptr);
}
std::string HsaPlatform::name() {
  TRACE();
  return "HSA";
}
std::vector<Property> HsaPlatform::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> HsaPlatform::enumerate() {
  TRACE();

  std::vector<std::tuple<bool, uint32_t, hsa_agent_t>> agents;
  CHECKED("Enumerate agents", //
          hsa_iterate_agents(
              [](hsa_agent_t agent, void *data) {
                hsa_device_type_t device_type;
                if (auto result = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
                    HSA_STATUS_SUCCESS == result) {
                  hsa_agent_feature_t features;
                  CHECKED("Query agent feature", hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features));
                  uint32_t queueSize;
                  CHECKED("Query agent queue size",
                          hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queueSize));
                  static_cast<decltype(agents) *>(data)->emplace_back( //
                      features & HSA_AGENT_FEATURE_KERNEL_DISPATCH, queueSize, agent);
                  return HSA_STATUS_SUCCESS;
                } else
                  return result;
              },
              &agents));

  auto host = std::find_if(agents.begin(), agents.end(), [](auto x) {
    auto &[dispatch, queueSize, _] = x;
    return queueSize == 0;
  });
  if (host == agents.end())
    throw std::logic_error(std::string(ERROR_PREFIX) + "No host agent available, AMDKFD not initialised?");
  hsa_agent_t hostAgent = std::get<2>(*host);

  std::vector<std::unique_ptr<Device>> devices;
  for (auto &[_, queueSize, agent] : agents) {
    if (agent.handle == hostAgent.handle) continue;
    devices.push_back(std::make_unique<HsaDevice>(queueSize, hostAgent, agent));
  }
  return devices;
}

// ---

HsaDevice::HsaDevice(uint32_t queueSize, hsa_agent_t hostAgent, hsa_agent_t agent)
    : queueSize(queueSize), hostAgent(hostAgent), agent(agent),
      store(
          ERROR_PREFIX,
          [this](auto &&s) {
            TRACE();
            hsa_code_object_reader_t reader;
            CHECKED("Load code object from memory",
                    hsa_code_object_reader_create_from_memory(s.data(), s.size(), &reader));
            hsa_executable_t executable;
            CHECKED("Create executable",
                    hsa_executable_create_alt(HSA_PROFILE_BASE, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR, nullptr,
                                              &executable));
            CHECKED("Load code object to executable with agent",
                    hsa_executable_load_agent_code_object(executable, this->agent, reader, nullptr, nullptr));
            CHECKED("Free executable", hsa_executable_freeze(executable, nullptr));
            uint32_t validationResult = 0;
            CHECKED("Validate executable", hsa_executable_validate(executable, &validationResult));
            if (validationResult != 0)
              throw std::logic_error(std::string(ERROR_PREFIX) +
                                     "Cannot validate executable: " + std::to_string(validationResult));
            CHECKED("Release code object reader", hsa_code_object_reader_destroy(reader));
            return executable;
          },
          [this](auto &&m, auto &&name) {
            TRACE();
            hsa_executable_symbol_t symbol;
            // HSA suffixes the entry with .kd, e.g. `theKernel` is `theKernel.kd`
            CHECKED("Resolve symbol",
                    hsa_executable_get_symbol_by_name(m, (name + ".kd").c_str(), &this->agent, &symbol));
            return symbol;
          },
          [&](auto &&m) {
            TRACE();
            CHECKED("Release executable", hsa_executable_destroy(m));
          },
          [&](auto &&) { TRACE(); }) {
  TRACE();
  // As per HSA_AGENT_INFO_NAME, name must be <= 63 chars
  deviceName =
      detail::allocateAndTruncate([&](auto &&data, auto) { hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, data); }, 64);

  CHECKED("Enumerate HSA agent regions", //
          hsa_agent_iterate_regions(
              agent,
              [](hsa_region_t region, void *data) {
                hsa_region_segment_t segment;
                hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
                if (segment != HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
                hsa_region_global_flag_t flags;
                hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
                if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
                  static_cast<decltype(this)>(data)->kernelArgRegion = region;
                }
                return HSA_STATUS_SUCCESS;
              },
              this));

  TRACE();

  CHECKED("Enumerate HSA AMD memory pools", //
          hsa_amd_agent_iterate_memory_pools(
              agent,
              [](hsa_amd_memory_pool_t pool, void *data) {
                hsa_amd_segment_t segment;
                hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
                if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

                bool accessibleByAll;
                hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &accessibleByAll);
                if (!accessibleByAll) {
                  static_cast<decltype(this)>(data)->deviceGlobalRegion = pool;
                }
                return HSA_STATUS_SUCCESS;
              },
              this));
  TRACE();

  if (kernelArgRegion.handle == 0) {
    throw std::logic_error(std::string(ERROR_PREFIX) + "No kernel arg region available form agent");
  }
  TRACE();

  if (deviceGlobalRegion.handle == 0) {
    throw std::logic_error(std::string(ERROR_PREFIX) + "No global device region available form agent");
  }
  TRACE();
}
HsaDevice::~HsaDevice() { TRACE(); }
int64_t HsaDevice::id() {
  TRACE();
  return 0;
}
std::string HsaDevice::name() {
  TRACE();
  return deviceName;
}
bool HsaDevice::sharedAddressSpace() {
  TRACE();
  return false;
}
std::vector<Property> HsaDevice::properties() {
  TRACE();
  return {};
}
void HsaDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  store.loadModule(name, image);
}
bool HsaDevice::moduleLoaded(const std::string &name) {
  TRACE();
  return store.moduleLoaded(name);
}
uintptr_t HsaDevice::malloc(size_t size, Access) {
  TRACE();
  if (size == 0) throw std::logic_error(std::string(ERROR_PREFIX) + "Cannot malloc size of 0");
  void *data;
  CHECKED("Allocate AMD memory pool", hsa_amd_memory_pool_allocate(deviceGlobalRegion, size, 0, &data));
  return reinterpret_cast<uintptr_t>(data);
}
void HsaDevice::free(uintptr_t ptr) {
  TRACE();
  CHECKED("Release AMD memory pool", hsa_amd_memory_pool_free(reinterpret_cast<void *>(ptr)));
}
std::unique_ptr<DeviceQueue> HsaDevice::createQueue() {
  TRACE();
  //  context.touch();
  hsa_queue_t *queue;
  CHECKED("Allocate agent queue", hsa_queue_create(agent, queueSize, HSA_QUEUE_TYPE_SINGLE, //
                                                   nullptr, nullptr,                        //
                                                   std::numeric_limits<uint32_t>::max(),    //
                                                   std::numeric_limits<uint32_t>::max(),    //
                                                   &queue));
  return std::make_unique<HsaDeviceQueue>(*this, queue);
}

// ---

HsaDeviceQueue::HsaDeviceQueue(decltype(device) device, decltype(queue) queue) : device(device), queue(queue) {
  TRACE();
  CHECKED("Allocate kernel signal", hsa_signal_create(1, 0, nullptr, &kernelSignal));
  CHECKED("Allocate H2D signal", hsa_signal_create(1, 0, nullptr, &hostToDeviceSignal));
  CHECKED("Allocate D2H signal", hsa_signal_create(1, 0, nullptr, &deviceToHostSignal));
}
HsaDeviceQueue::~HsaDeviceQueue() {
  TRACE();
  CHECKED("Release kernel signal", hsa_signal_destroy(kernelSignal));
  CHECKED("Release H2D signal", hsa_signal_destroy(hostToDeviceSignal));
  CHECKED("Release D2H signal", hsa_signal_destroy(deviceToHostSignal));
  CHECKED("Release agent queue", hsa_queue_destroy(queue));
}
void HsaDeviceQueue::enqueueCallback(hsa_signal_t &signal, const Callback &cb) {
  TRACE();
  CHECKED("Attach async handler to singnal", hsa_amd_signal_async_handler(
                                                 signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                 [](hsa_signal_value_t value, void *data) -> bool {
                                                   // Signals trigger when value is set to 0 or less, anything not 0 is
                                                   // an error.
                                                   CHECKED("Validate async signal value",
                                                           static_cast<hsa_status_t>(value));
                                                   detail::CountedCallbackHandler::consume(data);
                                                   return false;
                                                 },
                                                 detail::CountedCallbackHandler::createHandle(cb)));
}

void HsaDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  void *lockedHostSrcPtr;
  CHECKED("Lock host memory", hsa_amd_memory_lock(const_cast<void *>(src), size, nullptr, 0, &lockedHostSrcPtr));

  hsa_signal_store_relaxed(hostToDeviceSignal, 1);
  CHECKED("Copy memory async (H2D)",
          hsa_amd_memory_async_copy(reinterpret_cast<void *>(dst), device.agent, lockedHostSrcPtr, device.hostAgent, //
                                    size, 0, nullptr, hostToDeviceSignal));
  enqueueCallback(hostToDeviceSignal, [cb, lockedHostSrcPtr]() {
    CHECKED("Unlock host memory", hsa_amd_memory_unlock(lockedHostSrcPtr));
    if (cb) (*cb)();
  });

  //  CHECKED(hsa_memory_copy(reinterpret_cast<void *>(dst), src, size));
  //  if (cb) (*cb)();
}
void HsaDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  void *lockedHostDstPtr;
  CHECKED("Lock host memory", hsa_amd_memory_lock(dst, size, nullptr, 0, &lockedHostDstPtr));

  hsa_signal_store_relaxed(deviceToHostSignal, 1);
  CHECKED("Copy memory async (D2H)",
          hsa_amd_memory_async_copy(lockedHostDstPtr, device.hostAgent, reinterpret_cast<void *>(src), device.agent, //
                                    size, 0, nullptr, deviceToHostSignal));
  enqueueCallback(deviceToHostSignal, [cb, lockedHostDstPtr]() {
    CHECKED("Unlock host memory", hsa_amd_memory_unlock(lockedHostDstPtr));
    if (cb) (*cb)();
  });

  //  CHECKED(hsa_memory_copy(dst, reinterpret_cast<void *>(src), size));
  //  if (cb) (*cb)();
}
void HsaDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                        const std::vector<Type> &types, std::vector<void *> &args, const Policy &policy,
                                        const MaybeCallback &cb) {
  TRACE();
  if (types.back() != Type::Void)
    throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported");

  auto fn = device.store.resolveFunction(moduleName, symbol);
  uint64_t kernelObject;
  uint32_t kernargSegmentSize;
  uint32_t groupSegmentSize;
  uint32_t privateSegmentSize;
  CHECKED( //
      "Query executable kernel object",
      hsa_executable_symbol_get_info(fn, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject));
  CHECKED( //
      "Query executable kernel arg size",
      hsa_executable_symbol_get_info(fn, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernargSegmentSize));
  CHECKED( //
      "Query executable group segment size",
      hsa_executable_symbol_get_info(fn, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &groupSegmentSize));
  CHECKED( //
      "Query executable private segment size",
      hsa_executable_symbol_get_info(fn, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &privateSegmentSize));

  void *kernargAddress{};

  CHECKED("Allocate kernel arg memory",
          hsa_memory_allocate(device.kernelArgRegion, kernargSegmentSize, &kernargAddress));
  // Just to make sure future factors don't break things.
  static_assert(std::is_same_v<std::decay_t<decltype(args)>::value_type, void *>);
  auto *data = reinterpret_cast<uint8_t *>(kernargAddress);
  // Last arg is the return, void assertion should have been done before this.
  size_t byteOffset = 0;
  for (size_t i = 0; i < args.size() - 1; ++i) {
    if (byteOffset >= kernargSegmentSize) {
      throw std::logic_error(std::string(ERROR_PREFIX) + "Argument size out of bound, kernel expects " +
                             std::to_string(kernargSegmentSize) + " bytes, argument " + std::to_string(i) +
                             " leads to overflow");
    }
    if (types[i] == Type::Void) throw std::logic_error(std::string(ERROR_PREFIX) + "Illegal argument type: void");
    size_t size = byteOfType(types[i]);
    std::memcpy(data + byteOffset, args[i], size);
    byteOffset += size;
  }

  auto grid = policy.global;
  auto block = policy.local.value_or(Dim3{});

  hsa_signal_store_relaxed(kernelSignal, 1);
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  const uint32_t queueMask = queue->size - 1;
  hsa_kernel_dispatch_packet_t *dispatch =
      &((static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address))[index & queueMask]);

  dispatch->setup |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  dispatch->workgroup_size_x = int_cast<uint16_t>(block.x);
  dispatch->workgroup_size_y = int_cast<uint16_t>(block.y);
  dispatch->workgroup_size_z = int_cast<uint16_t>(block.z);
  dispatch->grid_size_x = int_cast<uint32_t>(grid.x);
  dispatch->grid_size_y = int_cast<uint32_t>(grid.y);
  dispatch->grid_size_z = int_cast<uint32_t>(grid.z);
  dispatch->completion_signal = kernelSignal;
  dispatch->kernel_object = kernelObject;
  dispatch->kernarg_address = kernargAddress;
  dispatch->private_segment_size = privateSegmentSize;
  dispatch->group_segment_size = groupSegmentSize;

  uint16_t header = 0;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;

  // XXX not entirely sure why the header needs to be done like this but not anything else
  // See:
  // https://github.com/HSAFoundation/HSA-Runtime-AMD/blob/0579a4f41cc21a76eff8f1050833ef1602290fcc/sample/vector_copy.c#L323
  std::atomic_ref<uint16_t> headerRef(header);
  headerRef.store(header, std::memory_order_release);

  hsa_queue_store_write_index_relaxed(queue, index + 1);
  hsa_signal_store_relaxed(queue->doorbell_signal, static_cast<hsa_signal_value_t>(index));

  enqueueCallback(kernelSignal, [cb, kernargAddress]() {
    if (cb) (*cb)();
    CHECKED("Release kernel arg memory", hsa_memory_free(kernargAddress));
  });
}

#undef CHECKED
