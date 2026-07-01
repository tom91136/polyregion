#include <atomic>
#include <chrono>
#include <thread>

#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/device_lock.h"
#include "polyinvoke/runtime.h"

#include "kernels/generated_cpu_reloc_fma.hpp"
#include "kernels/generated_cpu_shared_fma.hpp"
#include "kernels/generated_cuda_fma.hpp"
#include "kernels/generated_hsa_fma.hpp"
#include "kernels/generated_metal_fma.hpp"
#include "kernels/generated_opencl_source_fma.hpp"
#include "kernels/generated_opencl_spirv_fma.hpp"
#include "kernels/generated_vulkan_fma.hpp"
#include "test_utils.h"

using namespace polyregion::invoke;
using namespace polyregion::test_utils;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;
using namespace std::chrono_literals;

namespace {

void runQueueLifetime(Context &ctx, Backend backend, Platform &, Device &device, const ImageGroup &imageGroup) {
  if (imageGroup.size() != 1) {
    POLYTEST_FAIL(ctx, "expected exactly 1 image for fma kernel, got {} ({})", imageGroup.size(), magic_enum::enum_name(backend));
  }

  std::string module_, function_;
  if (device.singleEntryPerModule()) {
    module_ = "fma";
    function_ = "main";
  } else {
    module_ = "module";
    function_ = "_fma";
  }
  device.loadModule(module_, imageGroup[0].second);

  auto out_d = device.template mallocDeviceTyped<float>(1, Access::RW);
  auto queue = device.createQueue(std::chrono::seconds(30));

  std::atomic<bool> cb_started{false};
  std::atomic<bool> cb_finished{false};

  float a = 1.f, b = 2.f, c = 3.f;
  ArgBuffer buffer;
  if (device.sharedAddressSpace()) buffer.append(Type::IntS64, nullptr);
  buffer.append({{Type::Float32, &a}, {Type::Float32, &b}, {Type::Float32, &c}, {Type::Ptr, &out_d}, {Type::Void, {}}});

  queue->enqueueInvokeAsync(module_, function_, buffer, {}, Callback{[&]() {
                              cb_started.store(true, std::memory_order_release);
                              std::this_thread::sleep_for(200ms);
                              cb_finished.store(true, std::memory_order_release);
                            }});

  const auto poll_deadline = std::chrono::steady_clock::now() + 30s;
  while (!cb_started.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() > poll_deadline) POLYTEST_FAIL(ctx, "callback never started within 30s");
    std::this_thread::sleep_for(1ms);
  }

  const auto dtor_t0 = std::chrono::steady_clock::now();
  queue.reset();
  const auto dtor_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - dtor_t0).count();

  POLYTEST_CHECK_S(ctx, cb_finished.load(std::memory_order_acquire),
                   "queue dtor returned in {}ms while callback still in flight (backend={})", dtor_elapsed_ms,
                   magic_enum::enum_name(backend));

  // Narrow the post-cb UB window on a buggy build before tearing down the device.
  while (!cb_finished.load(std::memory_order_acquire))
    std::this_thread::sleep_for(10ms);
  std::this_thread::sleep_for(50ms);
  device.freeDevice(out_d);
}

std::vector<Task> discoverAll() {
  return discoverMatrix({
#ifndef __APPLE__
      {"queue-lifetime-cuda", generated::cuda::fma, {Backend::CUDA}, &runQueueLifetime},
      {"queue-lifetime-hsa", generated::hsa::fma, {Backend::HSA}, &runQueueLifetime},
      {"queue-lifetime-hip", generated::hsa::fma, {Backend::HIP}, &runQueueLifetime},
      {"queue-lifetime-opencl-source", generated::opencl_source::fma, {Backend::OpenCL}, &runQueueLifetime, skipHasSpirv},
      {"queue-lifetime-opencl-spirv", generated::opencl_spirv::fma, {Backend::OpenCL}, &runQueueLifetime, skipNoSpirv},
      {"queue-lifetime-vulkan", generated::vulkan::fma, {Backend::Vulkan}, &runQueueLifetime},
      {"queue-lifetime-levelzero", generated::opencl_spirv::fma, {Backend::LevelZero}, &runQueueLifetime},
#endif
#ifdef RUNTIME_ENABLE_METAL
      {"queue-lifetime-metal", generated::metal::fma, {Backend::Metal}, &runQueueLifetime},
#endif
      {"queue-lifetime-cpu-reloc", generated::cpu_reloc::fma, {Backend::RelocatableObject}, &runQueueLifetime},
      {"queue-lifetime-cpu-shared", generated::cpu_shared::fma, {Backend::SharedObject}, &runQueueLifetime},
  });
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
