#pragma once

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "polystl/polystl.h"

template <typename F> //
std::invoke_result_t<F> __polyregion_offload_f1__(F f) {
  static bool offload = !std::getenv("POLYSTL_NO_OFFLOAD");
  std::invoke_result_t<F> result{};
  size_t totalObjects = 0;
  if (offload) {
    {
      const auto kernel = [&result, &f](const int64_t tid) { result = f(); };
      auto &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::HostThreaded>(kernel);
      std::byte argData[sizeof(decltype(kernel))];
      std::memcpy(argData, &kernel, sizeof(decltype(kernel)));
      for (auto &object : bundle.objects) {
        totalObjects++;
        if (__polyregion_dispatch_hostthreaded(1, &argData, bundle.moduleName, object)) return result;
      }
    }
    {
      const auto kernel = [&result, &f]() { result = f(); };
      auto &bundle = __polyregion_offload__<polyregion::runtime::PlatformKind::Managed>(kernel);
      for (auto &object : bundle.objects) {
        totalObjects++;
        if (__polyregion_dispatch_managed(1, 0, 0, sizeof(decltype(kernel)), &kernel, bundle.moduleName, object)) return result;
      }
    }
    throw std::logic_error("Dispatch failed: no compatible backend after trying " + std::to_string(totalObjects) + " different objects");
  } else {
    [&result, &f]() { result = f(); }();
    return result;
  }
}