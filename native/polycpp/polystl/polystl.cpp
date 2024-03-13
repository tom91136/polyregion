#include "polystl/polystl.h"
#include "concurrency_utils.hpp"
#include "polyrt/runtime.h"

using namespace polyregion::runtime;

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

static std::unique_ptr<polyregion::runtime::Platform> createPlatform() {
  if (auto env = std::getenv("POLY_PLATFORM"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    if (auto it = NameToBackend.find(name); it != NameToBackend.end()) return Platform::of(it->second);
  }
  return {};
}

static std::unique_ptr<polyregion::runtime::Device> selectDevice(polyregion::runtime::Platform &p) {
  auto devices = p.enumerate();
  if (auto env = std::getenv("POLY_DEVICE"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    errno = 0;
    size_t index = std::strtol(name.c_str(), nullptr, 10);
    if (errno == 0 && index < devices.size()) { // we got a number, check inbounds and select device
      return std::move(devices.at(index));
    } else if (auto matching = // or do a substring match
               std::find_if(devices.begin(), devices.end(),
                            [&name](const auto &device) { return device->name().find(name) != std::string::npos; });
               matching != devices.end()) {
      return std::move(*matching);
    }
  } else if (!devices.empty()) return std::move(devices[0]);
  return {};
}

std::unique_ptr<Platform> polystl::thePlatform;
std::unique_ptr<Device> polystl::theDevice;
std::unique_ptr<DeviceQueue> polystl::theQueue;

void polystl::initialiseRuntime() {
  using namespace polystl;
  if (!thePlatform) {
    fprintf(stderr, "[POLYSTL] Initialising backends...\n");
    thePlatform = createPlatform();
    theDevice = thePlatform ? selectDevice(*thePlatform) : std::unique_ptr<Device>{};
    theQueue = theDevice ? theDevice->createQueue() : std::unique_ptr<DeviceQueue>{};
    if (thePlatform) {
      fprintf(stderr, "[POLYSTL] - Platform:%s [%s, %s]\n",
              thePlatform->name().c_str(),                  //
              to_string(thePlatform->kind()).data(),        //
              to_string(thePlatform->moduleFormat()).data() //
      );
    }
  }
}

static bool loadKernelObject(const polystl::KernelObject &object) {
  using namespace polystl;
  initialiseRuntime();
  if (!thePlatform || !theDevice || !theQueue) {
    fprintf(stderr, "[POLYSTL] No device/queue\n");
    return false;
  }

  if (thePlatform->kind() != object.kind || thePlatform->moduleFormat() != object.format) {
    fprintf(stderr, "[POLYSTL] Incompatible image  %s [%s] (targeting %s [%s])\n",
            to_string(object.kind).data(), //
            to_string(object.format).data(),
            to_string(thePlatform->kind()).data(), //
            to_string(thePlatform->moduleFormat()).data());
    return false;
  } //

  fprintf(stderr, "[POLYSTL] Loading image  %s [%s] (targeting %s [%s])\n",
          to_string(object.kind).data(), //
          to_string(object.format).data(),
          to_string(thePlatform->kind()).data(), //
          to_string(thePlatform->moduleFormat()).data());
  if (!theDevice->moduleLoaded(object.moduleName)) //
    theDevice->loadModule(object.moduleName, object.moduleImage);
  return true;
}

std::optional<polystl::PlatformKind> polystl::platformKind() {
  using namespace polystl;
  initialiseRuntime();
  if (!thePlatform || !theDevice || !theQueue) {
    fprintf(stderr, "[POLYSTL] No device/queue in %s\n", __func__);
    return {};
  }
  return thePlatform->kind();
}

bool polystl::dispatchHostThreaded(size_t global, void *functorData, const KernelObject &object) {
  if (!loadKernelObject(object)) return false;
  const static auto fn = __func__;
  polyregion::concurrency_utils::waitAll([&](auto &cb) {
    // FIXME why is the last arg (Void here, can be any type) needed?
    ArgBuffer buffer{{Type::Long64, nullptr}, {Type::Ptr, &functorData}, {Type::Void, nullptr}};
    theQueue->enqueueInvokeAsync(object.moduleName, "kernel", buffer, Policy{Dim3{global, 1, 1}}, [&]() {
      fprintf(stderr, "[POLYSTL:%s] Module %s completed\n", fn, object.moduleName.c_str());
      cb();
    });
  });

  std::fprintf(stderr, "[POLYSTL:%s] Done\n", fn);
  return true;
}

bool polystl::dispatchManaged(size_t global, size_t local, size_t localMemBytes, size_t functorDataSize, const void *functorData,
                              const KernelObject &object) {
  if (!loadKernelObject(object)) return false;
  const static auto fn = __func__;

  auto m = theDevice->mallocDevice(functorDataSize, polystl::Access::RW);

  polyregion::concurrency_utils::waitAll(
      [&](auto &cb) { polystl::theQueue->enqueueHostToDeviceAsync(functorData, m, functorDataSize, [&]() { cb(); }); });
  polyregion::concurrency_utils::waitAll([&](auto &cb) {
    ArgBuffer buffer{{Type::Ptr, &m}, {Type::Void, nullptr}};
    theQueue->enqueueInvokeAsync(
        object.moduleName, "kernel", buffer, //
        Policy{                              //
               Dim3{global, 1, 1},           //
               local > 0 ? std::optional{std::pair<Dim3, size_t>{Dim3{local, 0, 0}, localMemBytes}} : std::nullopt},
        [&]() {
          std::fprintf(stderr, "[POLYSTL:%s] Module %s completed\n", fn, object.moduleName.c_str());
          cb();
        });
  });
  theDevice->freeDevice(m);
  std::fprintf(stderr, "[POLYSTL:%s] Done\n", fn);
  return true;
}
