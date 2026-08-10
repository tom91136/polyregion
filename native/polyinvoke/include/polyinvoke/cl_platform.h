#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>

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

struct LaunchDimensions {
  Dim3 global;
  Dim3 local;
};

POLYREGION_EXPORT LaunchDimensions launchDimensions(const Dim3 &groups, const Dim3 &local);
POLYREGION_EXPORT std::optional<LaunchDimensions> retryLaunchDimensions(cl_int error, const Dim3 &groups, const Dim3 &local,
                                                                        size_t kernelMax);
POLYREGION_EXPORT std::string errorString(cl_int error);

struct SVMTracker {
  enum class Ownership { Device, Host, InheritedHost };

  struct Entry {
    size_t size;
    Ownership ownership;
  };

private:
  mutable std::mutex mutex;
  std::map<uintptr_t, Entry> entries;
  std::set<uintptr_t> leakedHostMaps;
  bool freeReleasesMap = false;

  auto owner(uintptr_t ptr);
  auto owner(uintptr_t ptr) const;

public:
  using Map = std::function<cl_int(void *, size_t)>;
  using Unmap = std::function<cl_int(void *)>;

  void track(void *ptr, size_t size);
  void untrack(void *ptr);
  std::optional<cl_int> mapForHost(void *ptr, const Map &map);
  cl_int mapAllForHost(const Map &map);
  cl_int unmapAllForDevice(const Unmap &unmap);
  std::vector<void *> pointers() const;
  std::optional<Ownership> ownership(void *ptr) const;
  bool freeReleasesHostMap() const;
};
} // namespace details

// per-device workarounds resolved once from the device name at construction
struct DeviceQuirks {
  bool nativeTrig;    // route POLY_* trig to native_ (llvmpipe libclc JIT crashes on precise range-reduction)
  size_t overReadPad; // zeroed pad per allocation to absorb llvmpipe SIMD over-reads past a buffer end
};

class POLYREGION_EXPORT ClDevice final : public Device {

  detail::LazyDroppable<cl_device_id> device;
  detail::LazyDroppable<cl_context> context;
  std::string deviceName;
  DeviceQuirks quirks;
  ModuleFormat format;
  details::ClCreateProgramWithIL_fn ilCreateFn; // non-null iff format==SPIRV_Kernel
  // Present iff device advertises buffer SVM; value is the memflags to OR into clSVMAlloc
  // (0 for coarse-grain, CL_MEM_SVM_FINE_GRAIN_BUFFER for fine-grain).
  std::optional<cl_bitfield> svm;
  std::shared_ptr<details::SVMTracker> svmTracker;
  details::ClModuleStore store; // must be dropped before the device
  detail::MemoryObjects<cl_mem> memoryObjects;
  std::optional<std::vector<std::string>> cachedFeatures; // XXX features() probes via clBuildProgram; cache so we pay once.

  void trackSvm(void *p, size_t size);
  void untrackSvm(void *p);

public:
  explicit ClDevice(cl_device_id device, ModuleFormat format, details::ClCreateProgramWithIL_fn ilCreateFn, std::optional<cl_bitfield> svm,
                    const std::string &platformName);
  ~ClDevice() override;
  POLYREGION_EXPORT int64_t id() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT PhysicalDevice physicalDevice() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT bool sharedAddressSpace() override;
  POLYREGION_EXPORT PagingMode pagingMode() override;
  POLYREGION_EXPORT bool singleEntryPerModule() override;
  POLYREGION_EXPORT size_t maxThreadsPerBlock() override;
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
  std::function<detail::MemoryObjects<cl_mem>::Resolved(uintptr_t)> queryMemObject;
  size_t memBaseAddrAlign;
  std::string deviceName;
  std::optional<cl_bitfield> svm; // forwarded from ClDevice; when set, use SVM ops instead of cl_mem
  std::shared_ptr<details::SVMTracker> svmTracker;

  void enqueueCallback(const MaybeCallback &cb, cl_event event);
  bool mapSvmForHost(void *ptr);
  void unmapAllSvmForDevice();
  void mapAllSvmForHost();

public:
  POLYREGION_EXPORT ClDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store, decltype(queue) queue,
                                  decltype(queryMemObject) queryMemObject, size_t memBaseAddrAlign, std::string deviceName,
                                  std::optional<cl_bitfield> svm, std::shared_ptr<details::SVMTracker> svmTracker);
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
