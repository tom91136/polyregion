#include "polystl/polystl.h"
#include "concurrency_utils.hpp"
#include "polyrt/runtime.h"

using namespace polyregion::runtime;

std::unique_ptr<Platform> __polyregion_selected_platform{}; // NOLINT(*-reserved-identifier)
std::unique_ptr<Device> __polyregion_selected_device{};     // NOLINT(*-reserved-identifier)
std::unique_ptr<DeviceQueue> __polyregion_selected_queue{}; // NOLINT(*-reserved-identifier)

extern "C" inline void __polyregion_select_platform() { // NOLINT(*-reserved-identifier)
  static const std::unordered_map<std::string, Backend> NameToBackend = {
      {"ptx", Backend::CUDA},  //
      {"cuda", Backend::CUDA}, //

      {"amdgpu", Backend::HIP}, //
      {"hip", Backend::HIP},    //
      {"hsa", Backend::HSA},    //

      {"opencl", Backend::OpenCL}, //
      {"ocl", Backend::OpenCL},    //
      {"cl", Backend::OpenCL},     //

      {"vulkan", Backend::Vulkan}, //
      {"vk", Backend::Vulkan},     //

      {"metal", Backend::Metal}, //
      {"mtl", Backend::Metal},   //
      {"apple", Backend::Metal}, //

      {"host_so", Backend::SharedObject},   //
      {"host", Backend::RelocatableObject}, //
  };
  if (auto env = std::getenv("POLY_PLATFORM"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    if (auto it = NameToBackend.find(name); it != NameToBackend.end()) __polyregion_selected_platform = Platform::of(it->second);
  }
}

extern "C" inline void __polyregion_select_device(polyregion::runtime::Platform &p) { // NOLINT(*-reserved-identifier)
  auto devices = p.enumerate();
  if (auto env = std::getenv("POLY_DEVICE"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    errno = 0;
    size_t index = std::strtol(name.c_str(), nullptr, 10);
    if (errno == 0 && index < devices.size()) { // we got a number, check inbounds and select device
      __polyregion_selected_device = std::move(devices.at(index));
    } else if (auto matching = // or do a substring match
               std::find_if(devices.begin(), devices.end(),
                            [&name](const auto &device) { return device->name().find(name) != std::string::npos; });
               matching != devices.end()) {
      __polyregion_selected_device = std::move(*matching);
    }
  } else if (!devices.empty()) __polyregion_selected_device = std::move(devices[0]);
}

extern "C" void __polyregion_initialise_runtime() { // NOLINT(*-reserved-identifier)
  if (!__polyregion_selected_platform) {
    fprintf(stderr, "[POLYSTL] Initialising backends...\n");
    __polyregion_select_platform();
    if (__polyregion_selected_platform) __polyregion_select_device(*__polyregion_selected_platform);
    if (__polyregion_selected_device) __polyregion_selected_queue = __polyregion_selected_device->createQueue();
    if (__polyregion_selected_platform) {
      fprintf(stderr, "[POLYSTL] - Platform:%s [%s, %s]\n",
              __polyregion_selected_platform->name().c_str(),                  //
              to_string(__polyregion_selected_platform->kind()).data(),        //
              to_string(__polyregion_selected_platform->moduleFormat()).data() //
      );
    }
  }
}

extern "C" inline bool __polyregion_load_kernel_object(const std::string &moduleName,
                                                       const KernelObject &object) { // NOLINT(*-reserved-identifier)
  __polyregion_initialise_runtime();
  if (!__polyregion_selected_platform || !__polyregion_selected_device || !__polyregion_selected_queue) {
    fprintf(stderr, "[POLYSTL] No device/queue\n");
    return false;
  }

  if (__polyregion_selected_platform->kind() != object.kind || __polyregion_selected_platform->moduleFormat() != object.format) {
    fprintf(stderr, "[POLYSTL] Incompatible image  %s [%s] (targeting %s [%s])\n",
            to_string(object.kind).data(), //
            to_string(object.format).data(),
            to_string(__polyregion_selected_platform->kind()).data(), //
            to_string(__polyregion_selected_platform->moduleFormat()).data());
    return false;
  } //

  fprintf(stderr, "[POLYSTL] Loading image  %s [%s] (targeting %s [%s])\n",
          to_string(object.kind).data(), //
          to_string(object.format).data(),
          to_string(__polyregion_selected_platform->kind()).data(), //
          to_string(__polyregion_selected_platform->moduleFormat()).data());
  if (!__polyregion_selected_device->moduleLoaded(moduleName)) //
    __polyregion_selected_device->loadModule(moduleName, object.moduleImage);
  return true;
}

extern "C" bool __polyregion_platform_kind(PlatformKind &kind) { // NOLINT(*-reserved-identifier)
  __polyregion_initialise_runtime();
  if (!__polyregion_selected_platform || !__polyregion_selected_device || !__polyregion_selected_queue) {
    fprintf(stderr, "[POLYSTL] No device/queue in %s\n", __func__);
    return false;
  }
  kind = __polyregion_selected_platform->kind();
  return true;
}

extern "C" bool __polyregion_dispatch_hostthreaded( // NOLINT(*-reserved-identifier)
    size_t global, void *functorData, const std::string &moduleName, const KernelObject &object) {
  if (!__polyregion_load_kernel_object(moduleName, object)) return false;
  const static auto fn = __func__;
  polyregion::concurrency_utils::waitAll([&](auto &cb) {
    // FIXME why is the last arg (Void here, can be any type) needed?
    ArgBuffer buffer{{Type::Long64, nullptr}, {Type::Ptr, &functorData}, {Type::Void, nullptr}};
    __polyregion_selected_queue->enqueueInvokeAsync(moduleName, "kernel", buffer, Policy{Dim3{global, 1, 1}}, [&]() {
      fprintf(stderr, "[POLYSTL:%s] Module %s completed\n", fn, moduleName.c_str());
      cb();
    });
  });

  std::fprintf(stderr, "[POLYSTL:%s] Done\n", fn);
  return true;
}

extern "C" bool __polyregion_dispatch_managed( // NOLINT(*-reserved-identifier)
    size_t global, size_t local, size_t localMemBytes, size_t functorDataSize, const void *functorData, const std::string &moduleName,
    const KernelObject &object) {
  if (!__polyregion_load_kernel_object(moduleName, object)) return false;
  const static auto fn = __func__;

  auto m = __polyregion_selected_device->mallocDevice(functorDataSize, Access::RW);

  polyregion::concurrency_utils::waitAll(
      [&](auto &cb) { __polyregion_selected_queue->enqueueHostToDeviceAsync(functorData, m, functorDataSize, [&]() { cb(); }); });
  polyregion::concurrency_utils::waitAll([&](auto &cb) {
    ArgBuffer buffer{{Type::Ptr, &m}, {Type::Void, nullptr}};
    __polyregion_selected_queue->enqueueInvokeAsync(
        moduleName, "kernel", buffer, //
        Policy{                              //
               Dim3{global, 1, 1},           //
               local > 0 ? std::optional{std::pair<Dim3, size_t>{Dim3{local, 0, 0}, localMemBytes}} : std::nullopt},
        [&]() {
          std::fprintf(stderr, "[POLYSTL:%s] Module %s completed\n", fn, moduleName.c_str());
          cb();
        });
  });
  __polyregion_selected_device->freeDevice(m);
  std::fprintf(stderr, "[POLYSTL:%s] Done\n", fn);
  return true;
}
