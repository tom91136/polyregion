#include "hsa_platform.h"
#include "json.hpp"
#include "utils.hpp"
#include <atomic>
#include <cstring>

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/ObjectFile.h"

using namespace polyregion::runtime;
using namespace polyregion::runtime::hsa;

#define CHECKED(m, f) checked((f), m, __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[HSA error] ";

static void checked(hsa_status_t result, const char *msg, const char *file, int line) {
  if (result != HSA_STATUS_SUCCESS) {
    const char *string = nullptr;
    hsa_status_string(result, &string);
    throw std::logic_error(
        std::string(ERROR_PREFIX) + file + ":" + std::to_string(line) + ": " +
        (string ? std::string(string) : "(hsa_status_string returned NULL, code=" + std::to_string(result) + ")"));
  }
}

HsaPlatform::HsaPlatform() {
  TRACE();
  hsaew_open("libhsa-runtime64.so.1");
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

static SymbolArgOffsetTable extractSymbolArgOffsetTable(const std::string &image) {
  using namespace llvm::object;

  const auto extractGeneric = []<typename ELFT>(const ELFObjectFile<ELFT> &obj) {
    using ELFT_Note = typename ELFT::Note;
    const auto &file = obj.getELFFile();
    if (auto sections = file.sections(); auto e = sections.takeError())
      throw std::logic_error(std::string(ERROR_PREFIX) + "Cannot read ELF sections: " + toString(std::move(e)));
    else {
      for (const auto s : *sections) {
        if (s.sh_type != llvm::ELF::SHT_NOTE) continue;
        auto Err = llvm::Error::success();
        for (ELFT_Note note : file.notes(s, Err)) {
          if (note.getName() != "AMDGPU" || note.getType() != llvm::ELF::NT_AMDGPU_METADATA) continue;
          try {
            using namespace nlohmann;
            auto kernels =
                nlohmann::json::from_msgpack(note.getDesc().begin(), note.getDesc().end()).at("amdhsa.kernels");
            SymbolArgOffsetTable offsetTable(kernels.size());
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
            throw std::logic_error("Illegal AMDGPU METADATA in .note section (" + std::string(e.what()) + ")");
          }
        }
      }
      throw std::logic_error("ELF image does not contains AMDGPU METADATA in the .note section");
    }
  };
  if (auto object = ObjectFile::createObjectFile(llvm::MemoryBufferRef(llvm::StringRef(image), ""));
      auto e = object.takeError()) {
    throw std::logic_error("Cannot load module: " + toString(std::move(e)));
  } else {
    if (const ELFObjectFileBase *elfObj = llvm::dyn_cast<ELFObjectFileBase>(object->get())) {
      if (auto obj = llvm::dyn_cast<ELF32LEObjectFile>(elfObj)) return extractGeneric(*obj);
      if (auto obj = llvm::dyn_cast<ELF32BEObjectFile>(elfObj)) return extractGeneric(*obj);
      if (auto obj = llvm::dyn_cast<ELF64LEObjectFile>(elfObj)) return extractGeneric(*obj);
      if (auto obj = llvm::dyn_cast<ELF64BEObjectFile>(elfObj)) return extractGeneric(*obj);
      throw std::logic_error("Unrecognised ELF variant");
    } else
      throw std::logic_error("Object image TypeID " + std::to_string(object->get()->getType()) +
                             " is not in the ELF range");
  }
}

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

            try {
              return std::make_pair(executable, extractSymbolArgOffsetTable(s));
            } catch (const std::exception &e) {
              throw std::logic_error(std::string(ERROR_PREFIX) +
                                     "Cannot to extract AMDGPU METADATA from image: " + e.what());
            }
          },
          [this](auto &&m, auto &&name, auto) {
            TRACE();
            auto [exec, offsets] = m;
            hsa_executable_symbol_t symbol;
            // HSA suffixes the entry with .kd, e.g. `theKernel` is `theKernel.kd`
            CHECKED("Resolve symbol",
                    hsa_executable_get_symbol_by_name(exec, (name + ".kd").c_str(), &this->agent, &symbol));
            if (auto it = offsets.find(name); it != offsets.end()) return std::make_pair(symbol, it->second);
            else
              throw std::logic_error(std::string(ERROR_PREFIX) + "Cannot argument offset table for symbol `" + name +
                                     "`");
          },
          [&](auto &&m) {
            TRACE();
            auto [exec, offsets] = m;
            CHECKED("Release executable", hsa_executable_destroy(exec));
          },
          [&](auto &&) { TRACE(); }) {
  TRACE();
  // As per HSA_AGENT_INFO_NAME, name must be <= 63 chars

  auto marketingName = detail::allocateAndTruncate(
      [&](auto &&data, auto) {
        hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_PRODUCT_NAME), data);
      },
      64);
  auto gfxArch =
      detail::allocateAndTruncate([&](auto &&data, auto) { hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, data); }, 64);
  deviceName = marketingName + "(" + gfxArch + ")";

  CHECKED("Enumerate HSA agent kernarg region", //
          hsa_agent_iterate_regions(
              agent,
              [](hsa_region_t region, void *data) {
                hsa_region_segment_t segment;
                CHECKED("Get region info (HSA_REGION_INFO_SEGMENT)",
                        hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment));
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
bool HsaDevice::singleEntryPerModule() {
  TRACE();
  return false;
}
std::vector<Property> HsaDevice::properties() {
  TRACE();
  return {};
}
std::vector<std::string> HsaDevice::features() {
  TRACE();
  auto gfxArch =
      detail::allocateAndTruncate([&](auto &&data, auto) { hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, data); }, 64);
  return {gfxArch};
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
}
HsaDeviceQueue::~HsaDeviceQueue() {
  TRACE();
  CHECKED("Release agent queue", hsa_queue_destroy(queue));
}

hsa_signal_t HsaDeviceQueue::createSignal(const char *message) {
  hsa_signal_t signal;
  CHECKED(message, hsa_signal_create(1, 0, nullptr, &signal));
  return signal;
}

void HsaDeviceQueue::destroySignal(const char *message, hsa_signal_t signal) {
  CHECKED(message, hsa_signal_destroy(signal));
}

void HsaDeviceQueue::enqueueCallback(hsa_signal_t signal, const Callback &cb) {
  CHECKED("Attach async handler to signal", //
          hsa_amd_signal_async_handler(
              signal, HSA_SIGNAL_CONDITION_LT, 1,
              [](hsa_signal_value_t value, void *data) -> bool {
                // Signals trigger when value is set to 0 or less, anything not 0 is an error.
                CHECKED("Validate async signal value", static_cast<hsa_status_t>(value));
                detail::CountedCallbackHandler::consume(data);
                return false;
              },
              detail::CountedCallbackHandler::createHandle(cb)));
}

void HsaDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  void *lockedHostSrcPtr;
  CHECKED("Lock host memory", hsa_amd_memory_lock(const_cast<void *>(src), size, nullptr, 0, &lockedHostSrcPtr));

  hsa_signal_t signal = createSignal("Allocate H2D signal");
  CHECKED("Copy memory async (H2D)",
          hsa_amd_memory_async_copy(reinterpret_cast<void *>(dst), device.agent, lockedHostSrcPtr, device.hostAgent, //
                                    size, 0, nullptr, signal));
  enqueueCallback(signal, [cb, lockedHostSrcPtr, signal, token = latch.acquire()]() {
    CHECKED("Unlock host memory", hsa_amd_memory_unlock(lockedHostSrcPtr));
    destroySignal("Release H2D signal", signal);
    if (cb) (*cb)();
  });
}

void HsaDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  void *lockedHostDstPtr;
  CHECKED("Lock host memory", hsa_amd_memory_lock(dst, size, nullptr, 0, &lockedHostDstPtr));

  hsa_signal_t signal = createSignal("Allocate D2H signal");
  CHECKED("Copy memory async (D2H)",
          hsa_amd_memory_async_copy(lockedHostDstPtr, device.hostAgent, reinterpret_cast<void *>(src), device.agent, //
                                    size, 0, nullptr, signal));
  enqueueCallback(signal, [cb, lockedHostDstPtr, signal, token = latch.acquire()]() {
    CHECKED("Unlock host memory", hsa_amd_memory_unlock(lockedHostDstPtr));
    destroySignal("Release H2D2H signal", signal);
    if (cb) (*cb)();
  });
}
void HsaDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                        const std::vector<Type> &types, std::vector<std::byte> argData, const Policy &policy,
                                        const MaybeCallback &cb) {
  TRACE();
  if (types.back() != Type::Void)
    throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported");

  auto [fn, argOffsets] = device.store.resolveFunction(moduleName, symbol, types);

  if (argOffsets.size() < types.size() - 1) {
    throw std::logic_error(std::string(ERROR_PREFIX) + "Symbol `" + symbol + "` expects at least " +
                           std::to_string(argOffsets.size()) + " arguments (excluding launch metadata) , " +
                           std::to_string(types.size()) + " given.");
  }

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

  if (kernargSegmentSize != 0) {
    CHECKED("Allocate kernel arg memory",
            hsa_memory_allocate(device.kernelArgRegion, kernargSegmentSize, &kernargAddress));
  }

  auto args = detail::argDataAsPointers(types, argData);
  // Just to make sure future refactors don't break things.
  static_assert(std::is_same_v<std::decay_t<decltype(args)>::value_type, void *>);

  if (!kernargAddress && args.size() - 1 != 0) {
    throw std::logic_error(std::string(ERROR_PREFIX) + "Kernarg address is NULL but we got " +
                           std::to_string(args.size() - 1) + " args to write");
  }
  auto *data = reinterpret_cast<uint8_t *>(kernargAddress);
  // Last arg is the return, void assertion should have been done before this.
  for (size_t i = 0; i < args.size() - 1; ++i) {
    if (argOffsets[i] >= kernargSegmentSize) {
      throw std::logic_error(std::string(ERROR_PREFIX) + "Argument size out of bound, kernel expects " +
                             std::to_string(kernargSegmentSize) + " bytes, argument " + std::to_string(i) +
                             " leads to overflow");
    }
    if (types[i] == Type::Void) throw std::logic_error(std::string(ERROR_PREFIX) + "Illegal argument type: void");
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
  hsa_kernel_dispatch_packet_t *dispatch =
      &((static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address))[index & queueMask]);

  dispatch->setup |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  dispatch->workgroup_size_x = int_cast<uint16_t>(block.x);
  dispatch->workgroup_size_y = int_cast<uint16_t>(block.y);
  dispatch->workgroup_size_z = int_cast<uint16_t>(block.z);
  dispatch->grid_size_x = int_cast<uint32_t>(grid.x);
  dispatch->grid_size_y = int_cast<uint32_t>(grid.y);
  dispatch->grid_size_z = int_cast<uint32_t>(grid.z);
  dispatch->completion_signal = signal;
  dispatch->kernel_object = kernelObject;
  dispatch->kernarg_address = kernargAddress;
  dispatch->private_segment_size = privateSegmentSize;
  dispatch->group_segment_size = std::max(groupSegmentSize, int_cast<uint32_t>(sharedMem));

  uint16_t header = 0;
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
  __atomic_store_n((uint16_t *)(&dispatch->header), header, __ATOMIC_RELEASE);
#endif

  hsa_queue_store_write_index_relaxed(queue, index + 1);
  hsa_signal_store_relaxed(queue->doorbell_signal, static_cast<hsa_signal_value_t>(index));
  TRACE();
  enqueueCallback(signal, [cb, kernargAddress, signal, token = latch.acquire()]() {
    CHECKED("Release kernel arg memory", hsa_memory_free(kernargAddress));
    destroySignal("Release kernel signal", signal);
    if (cb) (*cb)();
  });
}

#undef CHECKED
