#pragma once

#include <cstring>

#include "concurrency_utils.hpp"
#include "polyrt/runtime.h"

namespace polystl {

using namespace polyregion::runtime;

POLYREGION_EXPORT std::unique_ptr<polyregion::runtime::Platform> createPlatform();
POLYREGION_EXPORT std::unique_ptr<polyregion::runtime::Device> selectDevice(polyregion::runtime::Platform &p);

template <typename F>
POLYREGION_EXPORT void __polyregion_offload_dispatch__(size_t global,        //
                                                       size_t local,         //
                                                       size_t localMemBytes, //
                                                       F __f, const char *__kernelName, const unsigned char *__kernelImageBytes,
                                                       size_t __kernelImageSize) {

  char argData[sizeof(F)];
  std::memcpy(argData, (&__f), sizeof(F));
  void *argPtr = &argData;

  static auto thePlatform = createPlatform();
  static auto theDevice = thePlatform ? selectDevice(*thePlatform) : std::unique_ptr<polyregion::runtime::Device>{};
  static auto theQueue = theDevice ? theDevice->createQueue() : std::unique_ptr<polyregion::runtime::DeviceQueue>{};

  if (theDevice && theQueue) {
    if (!theDevice->moduleLoaded(__kernelName)) {
      theDevice->loadModule(__kernelName, std::string(__kernelImageBytes, __kernelImageBytes + __kernelImageSize));
    }
    polyregion::concurrency_utils::waitAll([&](auto &cb) {
      auto buffer = theDevice->leadingIndexArgument() //
                        ? ArgBuffer{{Type::Long64, nullptr}, {Type::Ptr, &argPtr}, {Type::Int32, nullptr}}
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
  } else {
    fprintf(stderr, "Host fallback\n");
    //    for (size_t i = 0; i < global; ++i) {
    //      __f(i);
    //    }
  }
}

} // namespace polystl
