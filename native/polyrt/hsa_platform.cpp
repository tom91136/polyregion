#include <atomic>
#include <cstring>

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"

#include "nlohmann/json.hpp"
#include "polyrt/hsa_platform.h"

using namespace polyregion::runtime;
using namespace polyregion::runtime::hsa;

#define CHECKED(m, f) checked((f), m, __FILE__, __LINE__)

static constexpr const char *PREFIX = "HSA";

static void checked(hsa_status_t result, const char *msg, const char *file, int line) {
  if (result != HSA_STATUS_SUCCESS) {
    const char *status = nullptr;
    hsa_status_string(result, &status);
    if (status) POLYRT_FATAL(PREFIX, "%s:%d %s [%s]", file, line, msg, status);
    else POLYRT_FATAL(PREFIX, "%s:%d %s [hsa_status_string returned NULL, code=%d]", file, line, msg, result);
  }
}
std::variant<std::string, std::unique_ptr<Platform>> HsaPlatform::create() {
  switch (hsaew_open("libhsa-runtime64.so.1")) {
    // both cases are fine, keep going
    case HSAEW_SUCCESS: break;
    case HSAEW_ALREADY_OPENED: break;
  }
  if (auto result = hsa_init(); result != HSA_STATUS_SUCCESS) {
    const char *status = nullptr;
    hsa_status_string(result, &status);
    return "HSA initialisation failed:" + (status ? std::string(status) : "(code=" + std::to_string(result) + ")");
  }
  return std::unique_ptr<Platform>(new HsaPlatform());
}
HsaPlatform::HsaPlatform() {
  POLYRT_TRACE();
  hsa_amd_register_system_event_handler(
      [](const hsa_amd_event_t *event, void *data) -> hsa_status_t {
        if (event->event_type == HSA_AMD_GPU_MEMORY_FAULT_EVENT) {
          fprintf(stderr, "[%s] Memory fault at 0x%lx. Reason:", PREFIX, static_cast<uintptr_t>(event->memory_fault.virtual_address));
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT) //
            fprintf(stderr, "Page not present or supervisor privilege. ");
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_READ_ONLY) //
            fprintf(stderr, "Write access to a read-only page. ");
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_NX) //
            fprintf(stderr, "Execute access to a page marked NX. ");
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_HOST_ONLY) //
            fprintf(stderr, "Host access only. ");
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_DRAMECC) //
            fprintf(stderr, "ECC failure (if supported by HW). ");
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_IMPRECISE) //
            fprintf(stderr, "Can't determine the exact fault address. ");
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_SRAMECC) //
            fprintf(stderr, "SRAM ECC failure (ie registers, no fault address). ");
          if (event->memory_fault.fault_reason_mask & HSA_AMD_MEMORY_FAULT_HANG) //
            fprintf(stderr, "GPU reset following unspecified hang. ");
          fprintf(stderr, "\n");
          return HSA_STATUS_ERROR;
        }
        return HSA_STATUS_SUCCESS;
      },
      nullptr);
}
std::string HsaPlatform::name() {
  POLYRT_TRACE();
  return "HSA";
}
std::vector<Property> HsaPlatform::properties() {
  POLYRT_TRACE();
  return {};
}
PlatformKind HsaPlatform::kind() {
  POLYRT_TRACE();
  return PlatformKind::Managed;
}
ModuleFormat HsaPlatform::moduleFormat() {
  POLYRT_TRACE();
  return ModuleFormat::HSACO;
}
std::vector<std::unique_ptr<Device>> HsaPlatform::enumerate() {
  POLYRT_TRACE();

  std::vector<std::tuple<bool, uint32_t, hsa_agent_t>> agents;
  CHECKED("Enumerate agents", //
          hsa_iterate_agents(
              [](hsa_agent_t agent, void *data) {
                hsa_device_type_t device_type;
                if (auto result = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type); HSA_STATUS_SUCCESS == result) {
                  hsa_agent_feature_t features;
                  CHECKED("Query agent feature", hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features));
                  uint32_t queueSize;
                  CHECKED("Query agent queue size", hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queueSize));
                  static_cast<decltype(agents) *>(data)->emplace_back( //
                      features & HSA_AGENT_FEATURE_KERNEL_DISPATCH, queueSize, agent);
                  return HSA_STATUS_SUCCESS;
                } else return result;
              },
              &agents));

  auto host = std::find_if(agents.begin(), agents.end(), [](auto x) {
    auto &[dispatch, queueSize, _] = x;
    return queueSize == 0;
  });
  if (host == agents.end()) POLYRT_FATAL(PREFIX, "No host agent available, AMDKFD not initialised (total agents=%zu)", agents.size());
  hsa_agent_t hostAgent = std::get<2>(*host);

  std::vector<std::unique_ptr<Device>> devices;
  for (auto &[_, queueSize, agent] : agents) {
    if (agent.handle == hostAgent.handle) continue;
    devices.push_back(std::make_unique<HsaDevice>(queueSize, hostAgent, agent));
  }
  return devices;
}

// ---

template <typename ELFT> static auto extractGeneric(const llvm::object::ELFObjectFile<ELFT> &obj) {
  using ELFT_Note = typename ELFT::Note;
  const auto &file = obj.getELFFile();
  if (auto sections = file.sections(); auto e = sections.takeError())
    POLYRT_FATAL(PREFIX, "Cannot read ELF sections: %s", toString(std::move(e)).c_str());
  else {
    for (const auto s : *sections) {
      if (s.sh_type != llvm::ELF::SHT_NOTE) continue;
      auto Err = llvm::Error::success();
      for (ELFT_Note note : file.notes(s, Err)) {
        if (note.getName() != "AMDGPU" || note.getType() != llvm::ELF::NT_AMDGPU_METADATA) continue;
        try {
          using namespace nlohmann;
          auto kernels =
              nlohmann::json::from_msgpack(note.getDesc(s.sh_addralign).begin(), note.getDesc(s.sh_addralign).end()).at("amdhsa.kernels");
          details::SymbolArgOffsetTable offsetTable(kernels.size());
          for (auto &kernel : kernels) {
            auto kernelName = kernel.at(".name").template get<std::string>();
            auto args = kernel.at(".args");
            std::vector<size_t> offsets(args.size());
            for (size_t i = 0; i < offsets.size(); ++i)
              offsets[i] = args.at(i).at(".offset");
            offsetTable.emplace(kernelName, offsets);
          }
          return offsetTable;
        } catch (const std::exception &e) {
          POLYRT_FATAL(PREFIX, "Illegal AMDGPU METADATA in .note section (%s)", e.what());
        }
      }
    }
    POLYRT_FATAL(PREFIX, "ELF image does not contains AMDGPU METADATA in the .note section (sections=%zu)", sections->size());
  }
};

static details::SymbolArgOffsetTable extractSymbolArgOffsetTable(const std::string &image) {
  using namespace llvm::object;

  if (auto object = ObjectFile::createObjectFile(llvm::MemoryBufferRef(llvm::StringRef(image), "")); auto e = object.takeError()) {
    POLYRT_FATAL(PREFIX, "Cannot load module: %s", toString(std::move(e)).c_str());
  } else {
    if (const ELFObjectFileBase *elfObj = llvm::dyn_cast<ELFObjectFileBase>(object->get())) {
      if (const auto obj = llvm::dyn_cast<ELF32LEObjectFile>(elfObj)) return extractGeneric(*obj);
      if (const auto obj = llvm::dyn_cast<ELF32BEObjectFile>(elfObj)) return extractGeneric(*obj);
      if (const auto obj = llvm::dyn_cast<ELF64LEObjectFile>(elfObj)) return extractGeneric(*obj);
      if (const auto obj = llvm::dyn_cast<ELF64BEObjectFile>(elfObj)) return extractGeneric(*obj);
      POLYRT_FATAL(PREFIX, "Unrecognised ELF variant: %u", elfObj->getType());
    } else POLYRT_FATAL(PREFIX, "Object image TypeID %d is not in the ELF range", object->get()->getType());
  }
}

HsaDevice::HsaDevice(uint32_t queueSize, hsa_agent_t hostAgent, hsa_agent_t agent)
    : queueSize(queueSize), hostAgent(hostAgent), agent(agent),
      store(
          PREFIX,
          [this](auto &&s) {
            POLYRT_TRACE();
            hsa_code_object_reader_t reader;
            CHECKED("Load code object from memory", hsa_code_object_reader_create_from_memory(s.data(), s.size(), &reader));
            hsa_executable_t executable;
            CHECKED("Create executable",
                    hsa_executable_create_alt(HSA_PROFILE_BASE, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR, nullptr, &executable));
            CHECKED("Load code object to executable with agent",
                    hsa_executable_load_agent_code_object(executable, this->agent, reader, nullptr, nullptr));
            CHECKED("Free executable", hsa_executable_freeze(executable, nullptr));
            uint32_t validationResult = 0;
            CHECKED("Validate executable", hsa_executable_validate(executable, &validationResult));
            if (validationResult != 0) POLYRT_FATAL(PREFIX, "Cannot validate executable, code=%d", validationResult);
            CHECKED("Release code object reader", hsa_code_object_reader_destroy(reader));

            try {
              return std::make_pair(executable, extractSymbolArgOffsetTable(s));
            } catch (const std::exception &e) {
              POLYRT_FATAL(PREFIX, "Cannot to extract AMDGPU METADATA from image: %s", e.what());
            }
          },
          [this](auto &&m, auto &&name, auto) {
            POLYRT_TRACE();
            auto [exec, offsets] = m;
            hsa_executable_symbol_t symbol;
            // HSA suffixes the entry with .kd, e.g. `theKernel` is `theKernel.kd`
            CHECKED("Resolve symbol", hsa_executable_get_symbol_by_name(exec, (name + ".kd").c_str(), &this->agent, &symbol));
            if (auto it = offsets.find(name); it != offsets.end()) return std::make_pair(symbol, it->second);
            else POLYRT_FATAL(PREFIX, "Cannot argument offset table for symbol `%s`", name.c_str());
          },
          [&](auto &&m) {
            POLYRT_TRACE();
            auto [exec, offsets] = m;
            CHECKED("Release executable", hsa_executable_destroy(exec));
          },
          [&](auto &&) { POLYRT_TRACE(); }) {
  POLYRT_TRACE();
  // As per HSA_AGENT_INFO_NAME, name must be <= 63 chars

  auto marketingName = detail::allocateAndTruncate(
      [&](auto &&data, auto) { hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_PRODUCT_NAME), data); }, 64);
  auto gfxArch = detail::allocateAndTruncate([&](auto &&data, auto) { hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, data); }, 64);
  deviceName = marketingName + "(" + gfxArch + ")";

  CHECKED("Enumerate HSA agent kernarg region", //
          hsa_agent_iterate_regions(
              agent,
              [](hsa_region_t region, void *data) {
                hsa_region_segment_t segment;
                CHECKED("Get region info (HSA_REGION_INFO_SEGMENT)", hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment));
                if (segment != HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
                hsa_region_global_flag_t flags;
                CHECKED("Get region info (HSA_REGION_INFO_GLOBAL_FLAGS)",
                        hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags));
                if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
                  static_cast<decltype(this)>(data)->kernelArgRegion = region;
                }
                return HSA_STATUS_SUCCESS;
              },
              this));

  POLYRT_TRACE();

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
  POLYRT_TRACE();

  if (kernelArgRegion.handle == 0) {
    POLYRT_FATAL(PREFIX, "No kernel arg region available form agent: handle=%lu", kernelArgRegion.handle);
  }
  POLYRT_TRACE();

  if (deviceGlobalRegion.handle == 0) {
    POLYRT_FATAL(PREFIX, "No global device region available form agent: handle=%lu", deviceGlobalRegion.handle);
  }
  POLYRT_TRACE();
}
HsaDevice::~HsaDevice() { POLYRT_TRACE(); }
int64_t HsaDevice::id() {
  POLYRT_TRACE();
  return 0;
}
std::string HsaDevice::name() {
  POLYRT_TRACE();
  return deviceName;
}
bool HsaDevice::sharedAddressSpace() {
  POLYRT_TRACE();
  return false;
}
bool HsaDevice::singleEntryPerModule() {
  POLYRT_TRACE();
  return false;
}
std::vector<Property> HsaDevice::properties() {
  POLYRT_TRACE();
  return {};
}
std::vector<std::string> HsaDevice::features() {
  POLYRT_TRACE();
  auto gfxArch = detail::allocateAndTruncate([&](auto &&data, auto) { hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, data); }, 64);
  return {gfxArch};
}
void HsaDevice::loadModule(const std::string &name, const std::string &image) {
  POLYRT_TRACE();
  store.loadModule(name, image);
}
bool HsaDevice::moduleLoaded(const std::string &name) {
  POLYRT_TRACE();
  return store.moduleLoaded(name);
}
uintptr_t HsaDevice::mallocDevice(size_t size, Access) {
  POLYRT_TRACE();
  if (size == 0) POLYRT_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  void *data;
  CHECKED("Allocate AMD memory pool", hsa_amd_memory_pool_allocate(deviceGlobalRegion, size, 0, &data));
  return reinterpret_cast<uintptr_t>(data);
}
void HsaDevice::freeDevice(uintptr_t ptr) {
  POLYRT_TRACE();
  CHECKED("Release AMD memory pool", hsa_amd_memory_pool_free(reinterpret_cast<void *>(ptr)));
}
std::optional<void *> HsaDevice::mallocShared(size_t size, Access access) {
  POLYRT_TRACE();
  if (size == 0) POLYRT_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  void *data;
  CHECKED("Allocate AMD memory pool", hsa_amd_memory_pool_allocate(deviceGlobalRegion, size, 0, &data));
  return data;
}
void HsaDevice::freeShared(void *ptr) {
  POLYRT_TRACE();
  CHECKED("Release AMD memory pool", hsa_amd_memory_pool_free(ptr));
}
std::unique_ptr<DeviceQueue> HsaDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYRT_TRACE();
  hsa_queue_t *queue;
  CHECKED("Allocate agent queue", hsa_queue_create(agent, queueSize, HSA_QUEUE_TYPE_SINGLE, //
                                                   nullptr, nullptr,                        //
                                                   std::numeric_limits<uint32_t>::max(),    //
                                                   std::numeric_limits<uint32_t>::max(),    //
                                                   &queue));
  return std::make_unique<HsaDeviceQueue>(timeout, *this, queue);
}

// ---

static hsa_signal_t createSignal(const char *message) {
  hsa_signal_t signal;
  CHECKED(message, hsa_signal_create(1, 0, nullptr, &signal));
  return signal;
}

static void destroySignal(const char *message, const hsa_signal_t signal) { CHECKED(message, hsa_signal_destroy(signal)); }

static void enqueueCallback(const hsa_signal_t signal, const Callback &cb) {
  static detail::CountedCallbackHandler handler;
  CHECKED("Attach async handler to signal", //
          hsa_amd_signal_async_handler(
              signal, HSA_SIGNAL_CONDITION_LT, 1,
              [](hsa_signal_value_t value, void *data) -> bool {
                // Signals trigger when value is set to 0 or less, anything not 0 is an error.
                CHECKED("Validate async signal value", static_cast<hsa_status_t>(value));
                handler.consume(data);
                return false;
              },
              handler.createHandle(cb)));
}

HsaDeviceQueue::HsaDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(device) device, decltype(queue) queue)
    : latch(timeout), device(device), queue(queue) {
  POLYRT_TRACE();
}
HsaDeviceQueue::~HsaDeviceQueue() {
  POLYRT_TRACE();
  CHECKED("Release agent queue", hsa_queue_destroy(queue));
}

void HsaDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYRT_TRACE();
  void *lockedHostSrcPtr;
  CHECKED("Lock host memory", hsa_amd_memory_lock(const_cast<void *>(src), size, nullptr, 0, &lockedHostSrcPtr));

  hsa_signal_t signal = createSignal("Allocate H2D signal");
  CHECKED("Copy memory async (H2D)",
          hsa_amd_memory_async_copy(reinterpret_cast<char *>(dst) + dstOffset, device.agent, lockedHostSrcPtr, device.hostAgent, //
                                    size, 0, nullptr, signal));

  enqueueCallback(signal, [cb, lockedHostSrcPtr, signal, token = latch.acquire()]() {
    CHECKED("Unlock host memory", hsa_amd_memory_unlock(lockedHostSrcPtr));
    destroySignal("Release H2D signal", signal);
    if (cb) (*cb)();
  });
  // XXX FIXME this wait makes the copy not actually async: switch to dependent signals to retain order...
  hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, std::numeric_limits<uint64_t>::max(), HSA_WAIT_STATE_BLOCKED);
}

void HsaDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYRT_TRACE();
  void *lockedHostDstPtr;
  CHECKED("Lock host memory", hsa_amd_memory_lock(dst, size, nullptr, 0, &lockedHostDstPtr));

  hsa_signal_t signal = createSignal("Allocate D2H signal");
  CHECKED("Copy memory async (D2H)",
          hsa_amd_memory_async_copy(lockedHostDstPtr, device.hostAgent, reinterpret_cast<char *>(src) + srcOffset, device.agent, //
                                    size, 0, nullptr, signal));

  enqueueCallback(signal, [cb, lockedHostDstPtr, signal, token = latch.acquire()]() {
    CHECKED("Unlock host memory", hsa_amd_memory_unlock(lockedHostDstPtr));
    destroySignal("Release D2H signal", signal);
    if (cb) (*cb)();
  });
  // XXX FIXME this wait makes the copy not actually async: switch to dependent signals to retain order...
  hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, std::numeric_limits<uint64_t>::max(), HSA_WAIT_STATE_BLOCKED);
}

void HsaDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                        std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYRT_TRACE();
  if (types.back() != Type::Void) POLYRT_FATAL(PREFIX, "Non-void return type not supported: %s", to_string(types.back()).data());

  auto [fn, argOffsets] = device.store.resolveFunction(moduleName, symbol, types);

  if (argOffsets.size() < types.size() - 1) {
    POLYRT_FATAL(PREFIX, "Symbol `%s` expects at least %ld arguments (excluding launch metadata), %ld given", symbol.c_str(),
                 argOffsets.size(), types.size());
  }

  uint64_t kernelObject;
  uint32_t kernargSegmentSize;
  uint32_t groupSegmentSize;
  uint32_t privateSegmentSize;
  CHECKED( //
      "Query executable kernel object", hsa_executable_symbol_get_info(fn, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject));
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

  if (kernargSegmentSize != 0) {
    CHECKED("Allocate kernel arg memory", hsa_memory_allocate(device.kernelArgRegion, kernargSegmentSize, &kernargAddress));
  }

  auto args = detail::argDataAsPointers(types, argData);
  // Just to make sure future refactors don't break things.
  static_assert(std::is_same_v<std::decay_t<decltype(args)>::value_type, void *>);

  if (!kernargAddress && args.size() - 1 != 0) {
    POLYRT_FATAL(PREFIX, "Kernarg address is NULL but we got %zu args to write", args.size() - 1);
  }
  auto *data = reinterpret_cast<uint8_t *>(kernargAddress);
  // Last arg is the return, void assertion should have been done before this.
  for (size_t i = 0; i < args.size() - 1; ++i) {
    if (argOffsets[i] >= kernargSegmentSize) {
      POLYRT_FATAL(PREFIX, "Argument size out of bound, kernel expects %u bytes, argument %ld leads to overflow", kernargSegmentSize, i);
    }
    if (types[i] == Type::Void) POLYRT_FATAL(PREFIX, "Illegal argument type: %s", to_string(types[i]).data());
    // XXX scratch (local) in HSA is a pointer of 4 bytes with the dynamic_shared_pointer kind
    size_t size = types[i] == Type::Scratch ? 4 : byteOfType(types[i]);
    if (types[i] == Type::Scratch) {
      // XXX In AMD ROCclr, the implementation of KernelBlitManager::setArgument writes the local memory size as an
      // argument here, but writing zero appears to have the same effect.
      std::memset(data + argOffsets[i], 0, size);
    } else {
      std::memcpy(data + argOffsets[i], args[i], size);
    }
  }
  auto [block, sharedMem] = policy.local.value_or(std::pair{Dim3{}, 0});
  auto grid = Dim3{policy.global.x * block.x, policy.global.y * block.y, policy.global.z * block.z};
  hsa_signal_t signal = createSignal("Allocate kernel signal");
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  const uint32_t queueMask = queue->size - 1;
  hsa_kernel_dispatch_packet_t *dispatch = &((static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address))[index & queueMask]);

  dispatch->setup |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  dispatch->workgroup_size_x = static_cast<uint16_t>(block.x);
  dispatch->workgroup_size_y = static_cast<uint16_t>(block.y);
  dispatch->workgroup_size_z = static_cast<uint16_t>(block.z);
  dispatch->grid_size_x = static_cast<uint32_t>(grid.x);
  dispatch->grid_size_y = static_cast<uint32_t>(grid.y);
  dispatch->grid_size_z = static_cast<uint32_t>(grid.z);
  dispatch->completion_signal = signal;
  dispatch->kernel_object = kernelObject;
  dispatch->kernarg_address = kernargAddress;
  dispatch->private_segment_size = privateSegmentSize;
  dispatch->group_segment_size = std::max(groupSegmentSize, static_cast<uint32_t>(sharedMem));

  uint16_t header = 0;
  header |= 1 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;

// XXX not entirely sure why the header needs to be done like this but not anything else
// See:
// https://github.com/HSAFoundation/HSA-Runtime-AMD/blob/0579a4f41cc21a76eff8f1050833ef1602290fcc/sample/vector_copy.c#L323
#ifdef _MSC_VER
  std::atomic_ref<uint16_t> headerRef(dispatch->header);
  headerRef.store(header, std::memory_order_release);
#else
  // XXX Many *nix systems doesn't implement atomic_ref yet, remove this in the future
  __atomic_store_n(&dispatch->header, header, __ATOMIC_RELEASE);
#endif

  hsa_queue_store_write_index_relaxed(queue, index + 1);
  hsa_signal_store_relaxed(queue->doorbell_signal, static_cast<hsa_signal_value_t>(index));
  POLYRT_TRACE();
  enqueueCallback(signal, [cb, kernargAddress, signal, token = latch.acquire()]() {
    CHECKED("Release kernel arg memory", hsa_memory_free(kernargAddress));
    destroySignal("Release kernel signal", signal);
    if (cb) (*cb)();
  });
}
void HsaDeviceQueue::enqueueWaitBlocking() {
  POLYRT_TRACE();
  latch.waitAll();
}

#undef CHECKED
