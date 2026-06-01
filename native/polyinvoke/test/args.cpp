#include <functional>
#include <vector>

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/device_lock.h"
#include "polyregion/concurrency_utils.hpp"

#include "kernels/generated_cpu_reloc_args.hpp"
#include "kernels/generated_cpu_shared_args.hpp"
#include "kernels/generated_cuda_args.hpp"
#include "kernels/generated_hsa_args.hpp"
#include "kernels/generated_metal_args.hpp"
#include "kernels/generated_opencl_source_args.hpp"
#include "kernels/generated_opencl_spirv_args.hpp"
#include "kernels/generated_vulkan_args.hpp"
#include "test_utils.h"

using namespace polyregion::invoke;
using namespace polyregion::compiletime;
using namespace polyregion::test_utils;
using namespace polyregion::concurrency_utils;
using namespace aspartame;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

void runArgs(Context &ctx, Backend backend, Platform &, Device &device, const ImageGroup &imageGroup) {
  std::function<std::string(size_t)> kernelName, moduleName;
  if (device.singleEntryPerModule()) {
    for (auto &[module_, data] : imageGroup)
      device.loadModule(module_, data);
    kernelName = [](auto) { return "main"; };
    moduleName = [](auto a) { return "arg" + std::to_string(a); };
  } else {
    if (imageGroup.size() != 1) {
      POLYTEST_FAIL(ctx, "expected exactly 1 image group, got {} for device `{}` (backend={})", //
                    imageGroup.size(), device.name(), magic_enum::enum_name(backend));
    }
    device.loadModule("module", imageGroup[0].second);
    kernelName = [](auto a) { return "_arg" + std::to_string(a); };
    moduleName = [](auto) { return "module"; };
  }

  auto q = device.createQueue(std::chrono::seconds(10));
  auto out_d = device.template mallocDeviceTyped<float>(1, Access::RW);

  for (int args = 0; args < 28; ++args) {
    float out = {};
    waitAll([&](auto &h) { q->enqueueHostToDeviceAsyncTyped(&out, out_d, 1, h); });
    const size_t scalarArgCount = args == 0 ? 0 : args - 1;
    auto values = iota(1.0f) | take(scalarArgCount) | to_vector();

    ArgBuffer buffer;
    if (device.sharedAddressSpace()) buffer.append(Type::IntS64, nullptr);
    for (auto &v : values)
      buffer.append(Type::Float32, &v);
    if (args != 0) buffer.append(Type::Ptr, &out_d);
    buffer.append(Type::Void, {});

    const float expected = args == 0 ? 0 : (scalarArgCount == 0 ? 42 : *(values ^ reduce(std::plus<>())));
    waitAll([&](auto &h) { q->enqueueInvokeAsync(moduleName(args), kernelName(args), buffer, {}, h); });
    waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1, h); });
    POLYTEST_CHECK_S(ctx, out == expected, "args={} actual={} expected={}", args, out, expected);
  }
  device.freeDevice(out_d);
}

std::vector<Task> discoverAll() {
  return discoverMatrix({
#ifndef __APPLE__
      {"args-cuda", generated::cuda::args, {Backend::CUDA}, &runArgs},
      {"args-hsa", generated::hsa::args, {Backend::HSA}, &runArgs},
      {"args-hip", generated::hsa::args, {Backend::HIP}, &runArgs},
      {"args-opencl-source", generated::opencl_source::args, {Backend::OpenCL}, &runArgs, skipHasSpirv},
      {"args-opencl-spirv", generated::opencl_spirv::args, {Backend::OpenCL}, &runArgs, skipNoSpirv},
      {"args-vulkan", generated::vulkan::args, {Backend::Vulkan}, &runArgs},
      {"args-levelzero", generated::opencl_spirv::args, {Backend::LevelZero}, &runArgs},
#endif
#ifdef RUNTIME_ENABLE_METAL
      {"args-metal", generated::metal::args, {Backend::Metal}, &runArgs},
#endif
      {"args-cpu-reloc", generated::cpu_reloc::args, {Backend::RelocatableObject}, &runArgs},
      {"args-cpu-shared", generated::cpu_shared::args, {Backend::SharedObject}, &runArgs},
  });
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
