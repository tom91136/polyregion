#pragma once

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "polyrt/rt.h"
#include "polystl/polystl.h"

template <typename F> //
std::invoke_result_t<F> __polyregion_offload_f1__(F f) {
  static bool offload = !std::getenv("POLYSTL_NO_OFFLOAD");
  std::invoke_result_t<F> result{};
  fprintf(stderr, "result=%p\n", &result);
  size_t totalObjects = 0;
  if (offload) {
    {
      auto kernel = [&result, &f](const int64_t tid) { result = f(); };
      auto &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::HostThreaded>(kernel);
      std::byte argData[sizeof(decltype(kernel))];
      std::memcpy(argData, &kernel, sizeof(decltype(kernel)));
      for (size_t i = 0; i < bundle.objectCount; ++i) {
        totalObjects++;
        if (polyregion::polyrt::loadKernelObject(bundle.moduleName, bundle.objects[i])) {
          polyregion::polyrt::dispatchHostThreaded(1, &argData, bundle.moduleName);
          return result;
        }
      }
    }
    {
      auto kernel = [  &result, f]() mutable { result = f(); };
      auto &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::Managed>(kernel);

      for (size_t i = 0; i < bundle.structCount; ++i) {
        if (i == bundle.interfaceLayoutIdx) fprintf(stderr, "**Exported**\n");
        bundle.structs[i].visualise(stderr);
      }

      if (bundle.structs[bundle.interfaceLayoutIdx].sizeInBytes != sizeof(decltype(kernel))) {
        throw std::logic_error("Exported TypeLayout size disagrees with size of kernel at compile time");
      }

      for (size_t i = 0; i < bundle.objectCount; ++i) {
        totalObjects++;
        if (polyregion::polyrt::loadKernelObject(bundle.moduleName, bundle.objects[i])) {
          polyregion::polystl::details::dispatchManaged(1, 0, 0, &bundle.structs[bundle.interfaceLayoutIdx], &kernel, bundle.moduleName);
          return result;
        }
      }
    }
    throw std::logic_error("Dispatch failed: no compatible backend after trying " + std::to_string(totalObjects) + " different objects");
  } else {
    [&result, &f]() { result = f(); }();
    return result;
  }
}