#include <atomic>
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

ImageGroups openClOffsetImages() {
  const std::string source = R"CLC(
kernel void offset_alias(global uchar *a_base, ulong a_offset,
                         global uchar *b_base, ulong b_offset,
                         global uchar *out_base, ulong out_offset,
                         global uchar *nullable_base, ulong nullable_offset) {
  global int *a = (global int *)(a_base + a_offset);
  global int *b = (global int *)(b_base + b_offset);
  global int *out = (global int *)(out_base + out_offset);
  global uchar *nullable = nullable_offset == (ulong)-1 ? (global uchar *)0 : nullable_base + nullable_offset;
  out[0] = nullable == 0 ? a[0] + b[0] : 1000;
}
kernel void compare_alias(global double *a_base, ulong a_offset,
                          global double *b_base, ulong b_offset,
                          global uint *out_base, ulong out_offset, uint n) {
  global double *a = (global double *)((global uchar *)a_base + a_offset);
  global double *b = (global double *)((global uchar *)b_base + b_offset);
  global uint *out = (global uint *)((global uchar *)out_base + out_offset);
  uint i = get_global_id(0);
  if (i < n) out[i] = (a + i) != (b + i);
}
kernel void legacy_pointer(global int *out) {
  out[0] = 73;
}
)CLC";
  return ImageGroups{{"", std::vector<uint8_t>(source.begin(), source.end())}};
}

void runOpenClOffsetArgs(Context &ctx, Backend, Platform &, Device &device, const ImageGroup &imageGroup) {
  POLYTEST_REQUIRE_S(ctx, imageGroup.size() == 1, "expected one OpenCL source image, got {}", imageGroup.size());
  device.loadModule("offset-module", imageGroup[0].second);
  auto q = device.createQueue(std::chrono::seconds(10));

  constexpr size_t bytes = 16384;
  std::vector<int> host(bytes / sizeof(int), 0);
  host[1] = 17;
  host[1025] = 25;
  const auto base = device.mallocDevice(bytes, Access::RW);
  waitAll([&](auto &h) { q->enqueueHostToDeviceAsync(host.data(), base, 0, bytes, h); });

  uintptr_t a = base + 4, b = base + 4100, out = base + 4, nullable = 0;
  ArgBuffer args{{Type::Ptr, &a}, {Type::Ptr, &b}, {Type::Ptr, &out}, {Type::Ptr, &nullable}, {Type::Void, nullptr}};
  waitAll([&](auto &h) { q->enqueueInvokeAsync("offset-module", "offset_alias", args, {}, h); });
  int actual = 0;
  waitAll([&](auto &h) { q->enqueueDeviceToHostAsync(base, 4, &actual, sizeof(actual), h); });
  POLYTEST_CHECK_S(ctx, actual == 42, "owner+offset pointer ABI produced {}, expected 42", actual);

  uint32_t count = 32;
  uintptr_t sameA = base, sameB = base, cmpOut = base + 8192;
  ArgBuffer cmpArgs{{Type::Ptr, &sameA}, {Type::Ptr, &sameB}, {Type::Ptr, &cmpOut}, {Type::IntU32, &count}, {Type::Void, nullptr}};
  waitAll(
      [&](auto &h) { q->enqueueInvokeAsync("offset-module", "compare_alias", cmpArgs, Policy{Dim3{1, 1, 1}, {{Dim3{256, 1, 1}, 0}}}, h); });
  std::vector<uint32_t> compared(count, 1);
  waitAll([&](auto &h) { q->enqueueDeviceToHostAsync(base, 8192, compared.data(), compared.size() * sizeof(uint32_t), h); });
  POLYTEST_CHECK_S(ctx, compared ^ forall([](uint32_t x) { return x == 0; }), "same-buffer pointer arguments compared unequal");

  uintptr_t legacyOut = base + 12288;
  ArgBuffer legacyArgs{{Type::Ptr, &legacyOut}, {Type::Void, nullptr}};
  waitAll([&](auto &h) { q->enqueueInvokeAsync("offset-module", "legacy_pointer", legacyArgs, {}, h); });
  int legacyActual = 0;
  waitAll([&](auto &h) { q->enqueueDeviceToHostAsync(base, 12288, &legacyActual, sizeof(legacyActual), h); });
  POLYTEST_CHECK_S(ctx, legacyActual == 73, "legacy pointer ABI produced {}, expected 73", legacyActual);

  std::atomic_bool callbackRan = false;
  q->enqueueInvokeAsync("offset-module", "offset_alias", args, {}, [&]() { callbackRan = true; });
  q.reset();
  POLYTEST_CHECK_S(ctx, callbackRan.load(), "queue destruction returned before its pending null-argument launch callback");
  device.freeDevice(base);
}

void runArgs(Context &ctx, Backend backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
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
  constexpr size_t interiorOffset = 4096;
  auto out_d = device.template mallocDeviceTyped<float>(interiorOffset / sizeof(float) + 1, Access::RW);
  auto outAt = out_d + interiorOffset;

  for (int args = 0; args < 28; ++args) {
    float out = {};
    waitAll([&](auto &h) { q->enqueueHostToDeviceAsyncTyped(&out, outAt, 1, h); });
    const size_t scalarArgCount = args == 0 ? 0 : args - 1;
    auto values = iota(1.0f) | take(scalarArgCount) | to_vector();

    ArgBuffer buffer;
    prependTidArg(platform, buffer);
    for (auto &v : values)
      buffer.append(Type::Float32, &v);
    if (args != 0) buffer.append(Type::Ptr, &outAt);
    buffer.append(Type::Void, {});

    const float expected = args == 0 ? 0 : (scalarArgCount == 0 ? 42 : *(values ^ reduce(std::plus<>())));
    waitAll([&](auto &h) { q->enqueueInvokeAsync(moduleName(args), kernelName(args), buffer, {}, h); });
    waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(outAt, &out, 1, h); });
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
      {"args-opencl-source-offset", openClOffsetImages(), {Backend::OpenCL}, &runOpenClOffsetArgs, skipHasSpirv},
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
