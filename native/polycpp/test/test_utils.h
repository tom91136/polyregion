#pragma once

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "polyregion/env_keys.h"
#include "polyrt/rt.h"
#include "polystl/algorithm_impl.h"
#include "polystl/polystl.h"

template <typename F> //
std::invoke_result_t<F> __polyregion_offload_f1__(F f) {
  static bool offload = !std::getenv(polyregion::env::PolystlNoOffload);
  std::invoke_result_t<F> result{};
  size_t totalObjects = 0;
  if (offload) {
    {
      auto kernel = [&result, &f](const int64_t) { result = f(); };
      auto &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::HostThreaded>(kernel);

      for (size_t i = 0; i < bundle.objectCount; ++i) {
        totalObjects++;
        if (polyregion::polyrt::loadKernelObject(bundle.moduleName, bundle.objects[i])) {
          polyregion::polystl::details::dispatchHostThreaded(1, &kernel, bundle.moduleName);
          return result;
        }
      }
    }
    {
      // XXX managed dispatch launches a whole workgroup; guard to lane 0 so a non-idempotent f runs once
      auto kernel = [&result, f]() mutable {
        if (__polyregion_builtin_gpu_global_idx(0) == 0) result = f();
      };
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
          void *kernelPtr = polyregion::polystl::details::polyreflectTrackPtr(&kernel);
          polyregion::polystl::details::dispatchManaged(1, 0, 0, &bundle.structs[bundle.interfaceLayoutIdx], kernelPtr, bundle.moduleName,
                                                        bundle.prelude, bundle.postlude);
          return result;
        }
      }
    }
    // no compatible image is a device capability gap (e.g. fp64), not a codegen bug: exit 77 (SKIP)
    (void)totalObjects;
    polyregion::polyrt::noCompatibleKernelExit("__polyregion_offload_f1__");
  } else {
    [&result, &f]() { result = f(); }();
    return result;
  }
}