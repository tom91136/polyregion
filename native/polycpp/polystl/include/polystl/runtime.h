#pragma once

#include <cstring>
#include <cstddef>

#include "concurrency_utils.hpp"
#include "polyrt/runtime.h"


uint64_t __polyregion__tid();

uint64_t __polyregion__gpu_global_idx(size_t);



namespace polystl {

struct KernelObject {
  size_t imageSize;
  const unsigned char* imageBytes;
  const char * kernelName;
  bool cpu;
  const char** features;

  // void * data;
  // ArgBuffer buffer;
};

using namespace polyregion::runtime;

POLYREGION_EXPORT std::unique_ptr<polyregion::runtime::Platform> createPlatform();
POLYREGION_EXPORT std::unique_ptr<polyregion::runtime::Device> selectDevice(polyregion::runtime::Platform &p);

static auto thePlatform = createPlatform();
static auto theDevice = thePlatform ? selectDevice(*thePlatform) : std::unique_ptr<polyregion::runtime::Device>{};
static auto theQueue = theDevice ? theDevice->createQueue() : std::unique_ptr<polyregion::runtime::DeviceQueue>{};

//POLYREGION_EXPORT void __polyregion_offload_dispatch__(const char**featu);

template <typename F>
POLYREGION_EXPORT bool dispatch(size_t global,        //
                                                       size_t local,         //
                                                       size_t localMemBytes, //
                                                       F __f, //
                                                       const char *__kernelName,
                                                       size_t __kernelImageSize,
                                                       const unsigned char *__kernelImageBytes
                                                       ) {

  char argData[sizeof(F)];
  std::memcpy(argData, (&__f), sizeof(F));
  void *argPtr = &argData;


  if (theDevice && theQueue) {
    if (!theDevice->moduleLoaded(__kernelName)) {
      theDevice->loadModule(__kernelName, std::string(__kernelImageBytes, __kernelImageBytes + __kernelImageSize));
    }
    polyregion::concurrency_utils::waitAll([&](auto &cb) {
      auto buffer = thePlatform->kind() == Platform::Kind::HostThreaded //
                        ? ArgBuffer{{Type::Long64, nullptr}, {Type::Ptr, &argPtr}, {Type::Void, nullptr}}
                        // FIXME why is the last int32 needed?
                        : ArgBuffer{{Type::Ptr, &argPtr}};

      theQueue->enqueueInvokeAsync(
          __kernelName, "kernel", buffer,
          Policy{                    //
                 Dim3{global, 1, 1}, //
                 local > 0 ? std::optional{std::pair<Dim3, size_t>{Dim3{local, 0, 0}, localMemBytes}} : std::nullopt},
          [&]() {
            fprintf(stderr, "Module %s completed\n", __kernelName);
            cb();
          });
    });
    fprintf(stderr, "Done\n");
    return true;
  } else {
    fprintf(stderr, "No device/queue\n");
     return false;
  }
}

} // namespace polystl
