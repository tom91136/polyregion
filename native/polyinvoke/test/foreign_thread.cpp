#include <thread>

#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/runtime.h"
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

void runForeignThread(Context &ctx, Backend backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
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
  auto q = device.createQueue(std::chrono::seconds(10));

  float out = {};
  std::thread([&] {
    auto out_d = device.template mallocDeviceTyped<float>(1, Access::RW);
    float a = 2.f, b = 3.f, c = 4.f;
    ArgBuffer buffer;
    prependTidArg(platform, buffer);
    buffer.append({{Type::Float32, &a}, {Type::Float32, &b}, {Type::Float32, &c}, {Type::Ptr, &out_d}, {Type::Void, {}}});
    waitAll([&](auto &h) { q->enqueueInvokeAsync(module_, function_, buffer, {}, h); });
    waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1, h); });
    device.freeDevice(out_d);
  }).join();
  POLYTEST_CHECK_S(ctx, approxEqual(out, 10.f), "fma from a thread other than the module's gave {} (backend={})", out,
                   magic_enum::enum_name(backend));
}

std::vector<Task> discoverAll() {
  return discoverMatrix({
#ifndef __APPLE__
      {"foreign-thread-cuda", generated::cuda::fma, {Backend::CUDA}, &runForeignThread},
      {"foreign-thread-hsa", generated::hsa::fma, {Backend::HSA}, &runForeignThread},
      {"foreign-thread-hip", generated::hsa::fma, {Backend::HIP}, &runForeignThread},
      {"foreign-thread-opencl-source", generated::opencl_source::fma, {Backend::OpenCL}, &runForeignThread, skipHasSpirv},
      {"foreign-thread-opencl-spirv", generated::opencl_spirv::fma, {Backend::OpenCL}, &runForeignThread, skipNoSpirv},
      {"foreign-thread-vulkan", generated::vulkan::fma, {Backend::Vulkan}, &runForeignThread},
      {"foreign-thread-levelzero", generated::opencl_spirv::fma, {Backend::LevelZero}, &runForeignThread},
#endif
#ifdef RUNTIME_ENABLE_METAL
      {"foreign-thread-metal", generated::metal::fma, {Backend::Metal}, &runForeignThread},
#endif
      {"foreign-thread-cpu-reloc", generated::cpu_reloc::fma, {Backend::RelocatableObject}, &runForeignThread},
      {"foreign-thread-cpu-shared", generated::cpu_shared::fma, {Backend::SharedObject}, &runForeignThread},
  });
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
