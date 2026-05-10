#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "polyregion/compat.h"

#include "clew.h"
#include "runtime.h"

namespace polyregion::invoke::cl {

class POLYREGION_EXPORT ClPlatform final : public Platform {
  POLYREGION_EXPORT explicit ClPlatform();

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~ClPlatform() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace details {
using ClModuleStore = detail::ModuleStore<cl_program, cl_kernel>;
using ClCreateProgramWithIL_fn = cl_program(CL_API_CALL *)(cl_context, const void *, size_t, cl_int *);
using ClSVMAlloc_fn = void *(CL_API_CALL *)(cl_context, cl_bitfield, size_t, cl_uint);
using ClSVMFree_fn = void(CL_API_CALL *)(cl_context, void *);
using ClEnqueueSVMMemcpy_fn = cl_int(CL_API_CALL *)(cl_command_queue, cl_bool, void *, const void *, size_t, cl_uint, const cl_event *,
                                                    cl_event *);
using ClSetKernelArgSVMPointer_fn = cl_int(CL_API_CALL *)(cl_kernel, cl_uint, const void *);
using ClSetKernelExecInfo_fn = cl_int(CL_API_CALL *)(cl_kernel, cl_uint, size_t, const void *);
using ClEnqueueSVMMap_fn = cl_int(CL_API_CALL *)(cl_command_queue, cl_bool, cl_bitfield, void *, size_t, cl_uint, const cl_event *,
                                                 cl_event *);
using ClEnqueueSVMUnmap_fn = cl_int(CL_API_CALL *)(cl_command_queue, void *, cl_uint, const cl_event *, cl_event *);

struct SVMFns {
  ClSVMAlloc_fn alloc;
  ClSVMFree_fn free;
  ClEnqueueSVMMemcpy_fn memcpy;
  ClSetKernelArgSVMPointer_fn setKernelArg;
  ClSetKernelExecInfo_fn setKernelExecInfo;
  ClEnqueueSVMMap_fn map;     // optional; null means caller must skip coarse-grain handshake
  ClEnqueueSVMUnmap_fn unmap; // optional; ditto
  cl_bitfield memFlags;       // CL_MEM_SVM_FINE_GRAIN_BUFFER, else 0 for coarse-grain
  explicit operator bool() const { return alloc && free && memcpy && setKernelArg && setKernelExecInfo; }
  bool coarseGrain() const { return memFlags == 0; }
};

// Coarse-grain SVM allocations require explicit clEnqueueSVMMap/clEnqueueSVMUnmap around host
// access. Track size + current map state here so the queue can flip allocs in/out around kernel
// dispatch. Fine-grain SVM ignores this (map/unmap are no-ops in spec, but we skip the calls).
struct SVMTracker {
  struct Entry {
    size_t size;
    bool mappedForHost; // true while host may safely read/write the alloc
  };
  std::mutex mtx;
  std::unordered_map<void *, Entry> entries;
};
} // namespace details

class POLYREGION_EXPORT ClDevice final : public Device {

  detail::LazyDroppable<cl_device_id> device;
  detail::LazyDroppable<cl_context> context;
  std::string deviceName;
  ModuleFormat format;
  details::ClCreateProgramWithIL_fn ilCreateFn; // non-null iff format==SPIRV_Kernel
  details::SVMFns svm{};                        // populated iff device advertises buffer SVM
  std::shared_ptr<details::SVMTracker> svmTracker;
  details::ClModuleStore store;                 // must be dropped before the device
  detail::MemoryObjects<cl_mem> memoryObjects;
  std::optional<std::vector<std::string>> cachedFeatures; // XXX features() probes via clBuildProgram; cache so we pay once.

public:
  explicit ClDevice(cl_device_id device, ModuleFormat format, details::ClCreateProgramWithIL_fn ilCreateFn, details::SVMFns svm);
  ~ClDevice() override;
  POLYREGION_EXPORT int64_t id() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT bool sharedAddressSpace() override;
  POLYREGION_EXPORT bool singleEntryPerModule() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT std::vector<std::string> features() override;
  POLYREGION_EXPORT void loadModule(const std::string &name, const std::string &image) override;
  POLYREGION_EXPORT bool moduleLoaded(const std::string &name) override;
  POLYREGION_EXPORT uintptr_t mallocDevice(size_t size, Access access) override;
  POLYREGION_EXPORT std::optional<void *> mallocShared(size_t size, Access access) override;
  POLYREGION_EXPORT void freeShared(void *ptr) override;
  POLYREGION_EXPORT void freeDevice(uintptr_t ptr) override;
  POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue(const std::chrono::duration<int64_t> &timeout) override;
};

class POLYREGION_EXPORT ClDeviceQueue final : public DeviceQueue {

  detail::CountingLatch latch;

  details::ClModuleStore &store;
  cl_command_queue queue = {};
  std::function<cl_mem(uintptr_t)> queryMemObject;
  details::SVMFns svm; // forwarded from ClDevice; when set, use SVM ops instead of cl_mem
  std::shared_ptr<details::SVMTracker> svmTracker;

  void enqueueCallback(const MaybeCallback &cb, cl_event event);
  // Coarse-grain SVM only: unmap all host-mapped allocs so the GPU can read/write them, then
  // remap them on completion so subsequent host accesses see the new contents. No-op if svm is
  // not coarse-grain or the impl lacks map/unmap.
  void unmapAllSvmForDevice();
  void mapAllSvmForHost();

public:
  POLYREGION_EXPORT ClDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store, decltype(queue) queue,
                                  decltype(queryMemObject) queryMemObject, details::SVMFns svm,
                                  std::shared_ptr<details::SVMTracker> svmTracker);
  POLYREGION_EXPORT ~ClDeviceQueue() override;
  POLYREGION_EXPORT void enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                    const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t bytes,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName,  //
                                            const std::string &symbol,      //
                                            const std::vector<Type> &types, //
                                            std::vector<std::byte> argData, //
                                            const Policy &policy, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueWaitBlocking() override;
};

} // namespace polyregion::invoke::cl
