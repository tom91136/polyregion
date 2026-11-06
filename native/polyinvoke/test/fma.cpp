#include <limits>
#include <vector>

#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/device_lock.h"
#include "polyinvoke/vulkan_platform.h"
#include "polyregion/concurrency_utils.hpp"

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
using namespace polyregion::concurrency_utils;
using polyregion::polytest::cases::approxEqual;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

const std::vector<float> xs{0.f,
                            -0.f,
                            1.f,
                            42.f,
                            -42.f,                                 //
                            std::numeric_limits<float>::epsilon(), //
                            std::numeric_limits<float>::max(),     //
                            std::numeric_limits<float>::min()};

void runFma(Context &ctx, Backend backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
  checkHostCapabilities(ctx, platform, device);
  if (imageGroup.size() != 1) {
    POLYTEST_FAIL(ctx, "expected exactly 1 image group, got {} for device `{}` (backend={})", //
                  imageGroup.size(), device.name(), magic_enum::enum_name(backend));
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

  auto q = device.createQueue(std::chrono::seconds(10));
  auto out_d = device.template mallocDeviceTyped<float>(1, Access::RW);

  for (auto a : xs)
    for (auto b : xs)
      for (auto c : xs) {
        ArgBuffer buffer;
        prependTidArg(platform, buffer);
        buffer.append({{Type::Float32, &a}, {Type::Float32, &b}, {Type::Float32, &c}, {Type::Ptr, &out_d}, {Type::Void, {}}});
        waitAll([&](auto &h) { q->enqueueInvokeAsync(module_, function_, buffer, {}, h); });
        float out = {};
        waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1, h); });
        const auto expected = a * b + c;
        // Subnormal-underflow corner: min*eps + 0 may flush to 0 on some HW; accept either.
        const bool acceptable = (c == 0.f
                                 && ((a == std::numeric_limits<float>::min() && b == std::numeric_limits<float>::epsilon())
                                     || (b == std::numeric_limits<float>::min() && a == std::numeric_limits<float>::epsilon())))
                                    ? (approxEqual(out, expected) || approxEqual(out, 0.f))
                                    : approxEqual(out, expected);
        POLYTEST_CHECK_S(ctx, acceptable, "a={} b={} c={} actual={} expected={}", a, b, c, out, expected);
      }
  device.freeDevice(out_d);
}

void runPipelineReuse(Context &ctx, Backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
  POLYTEST_REQUIRE_S(ctx, imageGroup.size() == 1, "expected exactly 1 image group, got {}", imageGroup.size());
  device.loadModule("fma", imageGroup[0].second);
  auto q = device.createQueue(std::chrono::seconds(10));
  auto *vulkanQueue = static_cast<vulkan::VulkanDeviceQueue *>(q.get());
  auto out_d = device.template mallocDeviceTyped<float>(1, Access::RW);
  const auto launch = [&](const size_t lanes) {
    float a = 2.f, b = 3.f, c = 4.f;
    ArgBuffer buffer;
    prependTidArg(platform, buffer);
    buffer.append({{Type::Float32, &a}, {Type::Float32, &b}, {Type::Float32, &c}, {Type::Ptr, &out_d}, {Type::Void, {}}});
    waitAll([&](auto &h) { q->enqueueInvokeAsync("fma", "main", buffer, Policy{{1, 1, 1}, {{{lanes, 1, 1}, 0}}}, h); });
    float out = {};
    waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1, h); });
    POLYTEST_CHECK_S(ctx, out == 10.f, "lanes={} actual={}", lanes, out);
  };
  for (int i = 0; i < 8; ++i)
    launch(1);
  POLYTEST_CHECK_S(ctx, vulkanQueue->pipelineCreations() == 1, "same local size created {} pipelines", vulkanQueue->pipelineCreations());
  launch(2);
  POLYTEST_CHECK_S(ctx, vulkanQueue->pipelineCreations() == 2, "new local size left {} pipelines", vulkanQueue->pipelineCreations());
  launch(1);
  launch(2);
  POLYTEST_CHECK_S(ctx, vulkanQueue->pipelineCreations() == 2, "revisited local sizes created {} pipelines",
                   vulkanQueue->pipelineCreations());
  device.freeDevice(out_d);
}

void runQueueSettle(Context &ctx, Backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
  POLYTEST_REQUIRE_S(ctx, imageGroup.size() == 1, "expected exactly 1 image group, got {}", imageGroup.size());
  const auto module_ = device.singleEntryPerModule() ? "fma" : "module";
  const auto function_ = device.singleEntryPerModule() ? "main" : "_fma";
  device.loadModule(module_, imageGroup[0].second);
  auto q = device.createQueue(std::chrono::seconds(10));
  auto out_d = device.template mallocDeviceTyped<float>(1, Access::RW);

  const bool copiesComplete = device.pagingMode() == PagingMode::None;
  float in = 1.5f, out = 0.f;
  q->enqueueHostToDeviceAsyncTyped(&in, out_d, 1);
  if (copiesComplete) POLYTEST_CHECK(ctx, q->quiescent());
  q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1);
  if (copiesComplete) POLYTEST_CHECK(ctx, q->quiescent());
  POLYTEST_CHECK_S(ctx, out == in, "copy round trip returned {}", out);

  float a = 2.f, b = 3.f, c = 4.f;
  ArgBuffer buffer;
  prependTidArg(platform, buffer);
  buffer.append({{Type::Float32, &a}, {Type::Float32, &b}, {Type::Float32, &c}, {Type::Ptr, &out_d}, {Type::Void, {}}});
  q->enqueueInvokeAsync(module_, function_, buffer, {}, {});
  POLYTEST_CHECK(ctx, !q->quiescent());
  q->settle();
  POLYTEST_CHECK(ctx, q->quiescent());
  q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1);
  q->settle();
  POLYTEST_CHECK_S(ctx, out == 10.f, "invoke result was {}", out);
  device.freeDevice(out_d);
}

std::vector<Task> discoverAll() {
  return discoverMatrix({
#ifndef __APPLE__
      {"fma-cuda", generated::cuda::fma, {Backend::CUDA}, &runFma},
      {"fma-hsa", generated::hsa::fma, {Backend::HSA}, &runFma},
      {"fma-hip", generated::hsa::fma, {Backend::HIP}, &runFma},
      {"fma-opencl-source", generated::opencl_source::fma, {Backend::OpenCL}, &runFma, skipHasSpirv},
      {"queue-settle-opencl-source", generated::opencl_source::fma, {Backend::OpenCL}, &runQueueSettle, skipHasSpirv},
      {"fma-opencl-spirv", generated::opencl_spirv::fma, {Backend::OpenCL}, &runFma, skipNoSpirv},
      {"fma-vulkan", generated::vulkan::fma, {Backend::Vulkan}, &runFma},
      {"pipeline-reuse-vulkan", generated::vulkan::fma, {Backend::Vulkan}, &runPipelineReuse},
      {"fma-levelzero", generated::opencl_spirv::fma, {Backend::LevelZero}, &runFma},
#endif
#ifdef RUNTIME_ENABLE_METAL
      {"fma-metal", generated::metal::fma, {Backend::Metal}, &runFma},
#endif
      {"fma-cpu-reloc", generated::cpu_reloc::fma, {Backend::RelocatableObject}, &runFma},
      {"fma-cpu-shared", generated::cpu_shared::fma, {Backend::SharedObject}, &runFma},
  });
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
