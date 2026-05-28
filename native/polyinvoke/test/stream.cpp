#include "polyregion/stream.hpp"

#include <memory>
#include <vector>

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/device_lock.h"

#include "kernels/generated_cpu_reloc_stream.hpp"
#include "kernels/generated_cpu_shared_stream.hpp"
#include "kernels/generated_cuda_stream_double.hpp"
#include "kernels/generated_cuda_stream_float.hpp"
#include "kernels/generated_hsa_stream_double.hpp"
#include "kernels/generated_hsa_stream_float.hpp"
#include "kernels/generated_metal_stream_float.hpp"
#include "kernels/generated_opencl_source_stream_double.hpp"
#include "kernels/generated_opencl_source_stream_float.hpp"
#include "kernels/generated_opencl_spirv_stream_double.hpp"
#include "kernels/generated_opencl_spirv_stream_float.hpp"
#include "kernels/generated_vulkan_stream.hpp"
#include "test_utils.h"

using namespace polyregion::invoke;
using namespace polyregion::test_utils;
using namespace aspartame;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

template <typename T>
void runStream(Context &ctx, Backend backend, Platform &platform, Device &device, const ImageGroup &imageGroup, Type tpe,
               const std::string &suffix, T relTolerance, const std::vector<size_t> &sizes, const std::vector<size_t> &groupSizes,
               const std::vector<size_t> &times) {
  polyregion::stream::Kernels<std::pair<std::string, std::string>> kernelSpecs;
  if (device.singleEntryPerModule()) {
    for (auto &[module_, data] : imageGroup)
      device.loadModule(module_, data);
    kernelSpecs = {.copy = {"stream_copy" + suffix, "main"},
                   .mul = {"stream_mul" + suffix, "main"},
                   .add = {"stream_add" + suffix, "main"},
                   .triad = {"stream_triad" + suffix, "main"},
                   .dot = {"stream_dot" + suffix, "main"}};
  } else {
    if (imageGroup.size() != 1) {
      POLYTEST_FAIL(ctx, "expected exactly 1 image group, got {} for device `{}` (backend={})", //
                    imageGroup.size(), device.name(), magic_enum::enum_name(backend));
    }
    device.loadModule("module", imageGroup[0].second);
    kernelSpecs = {.copy = {"module", "stream_copy" + suffix},
                   .mul = {"module", "stream_mul" + suffix},
                   .add = {"module", "stream_add" + suffix},
                   .triad = {"module", "stream_triad" + suffix},
                   .dot = {"module", "stream_dot" + suffix}};
  }

  for (size_t size : sizes)
    for (size_t time : times)
      for (size_t Ncore : groupSizes) {
        polyregion::stream::runStream<T>(
            tpe, size, time, Ncore, platform.name(), platform.kind(), device, kernelSpecs, /*verbose*/ false,
            [&](auto actual, auto limit) {
              POLYTEST_CHECK_S(ctx, actual < limit, "array size={} time={} Ncore={} actual={} limit={}", size, time, Ncore, actual, limit);
            },
            [&](auto actual) {
              POLYTEST_CHECK_S(ctx, actual < relTolerance, "dot size={} time={} Ncore={} actual={} tol={}", size, time, Ncore, actual,
                               relTolerance);
            });
      }
}

template <typename T>
DeviceRunner streamRunner(Type tpe, std::string suffix, T relTolerance, std::vector<size_t> sizes, std::vector<size_t> groupSizes,
                          std::vector<size_t> times) {
  return [tpe, suffix = std::move(suffix), relTolerance, sizes = std::move(sizes), groupSizes = std::move(groupSizes),
          times = std::move(times)](Context &ctx, Backend b, Platform &plat, Device &dev, const ImageGroup &img) {
    runStream<T>(ctx, b, plat, dev, img, tpe, suffix, relTolerance, sizes, groupSizes, times);
  };
}

const DeviceSkip skipHasSpirvOrNoFp64 = [](Backend b, Device &d) { return skipHasSpirv(b, d) || skipNoFp64(b, d); };
const DeviceSkip skipNoSpirvOrNoFp64 = [](Backend b, Device &d) { return skipNoSpirv(b, d) || skipNoFp64(b, d); };
// FIXME NVIDIA's Vulkan driver (595.71.05) SEGFAULTs inside libnvidia-glcore when a
// shader module is reused across pipelines with different specialization constant values
const DeviceSkip skipNvidiaVulkan = [](Backend, Device &d) { return d.features() | contains("nvidia"); };
const DeviceSkip skipNvidiaVulkanOrNoFp64 = [](Backend b, Device &d) { return skipNvidiaVulkan(b, d) || skipNoFp64(b, d); };

const std::vector<size_t> gpuGroups = {32, 64, 128, 256};
const std::vector<size_t> cpuGroups = {1, 2, 3, 4, 5, 6, 7, 8};
const std::vector<size_t> times = {1, 2, 10};
const std::vector<size_t> defaultSizes = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};

std::vector<Task> discoverAll() {
  auto floatRunner = streamRunner<float>(Type::Float32, "_float", 0.008f, defaultSizes, gpuGroups, times);
  auto doubleRunner = streamRunner<double>(Type::Float64, "_double", 0.008, defaultSizes, gpuGroups, times);
  auto vulkanFloatRunner = streamRunner<float>(Type::Float32, "_float", 0.008f, defaultSizes, {1, 2, 32, 64, 128, 256}, times);
  auto vulkanDoubleRunner = streamRunner<double>(Type::Float64, "_double", 0.008, {1024, 2048}, {1, 2, 32, 64}, times);
  auto cpuFloatRunner = streamRunner<float>(Type::Float32, "_float", 0.008f, defaultSizes, cpuGroups, times);
  auto cpuDoubleRunner = streamRunner<double>(Type::Float64, "_double", 0.0008, defaultSizes, cpuGroups, times);

  return discoverMatrix({
#ifndef __APPLE__
      {"stream-cuda-float", generated::cuda::stream_float, {Backend::CUDA}, floatRunner},
      {"stream-cuda-double", generated::cuda::stream_double, {Backend::CUDA}, doubleRunner, skipNoFp64},
      {"stream-hsa-float", generated::hsa::stream_float, {Backend::HSA}, floatRunner},
      {"stream-hsa-double", generated::hsa::stream_double, {Backend::HSA}, doubleRunner, skipNoFp64},
      {"stream-hip-float", generated::hsa::stream_float, {Backend::HIP}, floatRunner},
      {"stream-hip-double", generated::hsa::stream_double, {Backend::HIP}, doubleRunner, skipNoFp64},
      {"stream-opencl-source-float", generated::opencl_source::stream_float, {Backend::OpenCL}, floatRunner, skipHasSpirv},
      {"stream-opencl-source-double", generated::opencl_source::stream_double, {Backend::OpenCL}, doubleRunner, skipHasSpirvOrNoFp64},
      {"stream-opencl-spirv-float", generated::opencl_spirv::stream_float, {Backend::OpenCL}, floatRunner, skipNoSpirv},
      {"stream-opencl-spirv-double", generated::opencl_spirv::stream_double, {Backend::OpenCL}, doubleRunner, skipNoSpirvOrNoFp64},
      {"stream-vulkan-float", generated::vulkan::stream, {Backend::Vulkan}, vulkanFloatRunner, skipNvidiaVulkan},
      {"stream-vulkan-double", generated::vulkan::stream, {Backend::Vulkan}, vulkanDoubleRunner, skipNvidiaVulkanOrNoFp64},
      {"stream-levelzero-float", generated::opencl_spirv::stream_float, {Backend::LevelZero}, floatRunner},
      {"stream-levelzero-double", generated::opencl_spirv::stream_double, {Backend::LevelZero}, doubleRunner, skipNoFp64},
#endif
#ifdef RUNTIME_ENABLE_METAL
      {"stream-metal-float", generated::metal::stream_float, {Backend::Metal}, floatRunner},
#endif
      {"stream-cpu-reloc-float", generated::cpu_reloc::stream, {Backend::RelocatableObject}, cpuFloatRunner},
      {"stream-cpu-reloc-double", generated::cpu_reloc::stream, {Backend::RelocatableObject}, cpuDoubleRunner},
      {"stream-cpu-shared-float", generated::cpu_shared::stream, {Backend::SharedObject}, cpuFloatRunner},
      {"stream-cpu-shared-double", generated::cpu_shared::stream, {Backend::SharedObject}, cpuDoubleRunner},
  });
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
