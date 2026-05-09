#include <numeric>

#include "aspartame/all.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_range.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/concurrency_utils.hpp"
#include "polyregion/io.hpp"

#include "kernels/generated_cpu_args.hpp"
#include "kernels/generated_gpu_args.hpp"
#include "kernels/generated_msl_args.hpp"
#include "kernels/generated_spirv_glsl_args.hpp"
#include "test_utils.h"

using namespace polyregion::invoke;
using namespace polyregion::compiletime;
using namespace polyregion::test_utils;
using namespace polyregion::concurrency_utils;
using namespace aspartame;

// See https://github.com/JuliaGPU/AMDGPU.jl/issues/10

template <typename I> void testArgs(I images, std::initializer_list<Backend> backends) {
  std::vector<Backend> enabled;
  for (auto b : backends)
    if (!polyregion::test_utils::isBackendDisabled(b)) enabled.push_back(b);
  if (enabled.empty()) return;
  auto backend = GENERATE_COPY(from_range(enabled));
  auto platform = polyregion::test_utils::makePlatform(backend);
  DYNAMIC_SECTION("backend=" << magic_enum::enum_name(backend)) {
    for (auto &d : platform->enumerate()) {
      if (polyregion::test_utils::isDeviceDisabled(d->name())) continue;
      // XXX skip OpenCL SPIR-V-format duplicate; polyinvoke ships only source kernels.
      const auto df = d->features();
      if (backend == Backend::OpenCL && std::find(df.begin(), df.end(), "spirv_kernel") != df.end()) continue;
      DYNAMIC_SECTION("device=" << d->name()) {
        if (auto imageGroups = findTestImage(images, backend, d->features()); !imageGroups.empty()) {

          std::function<std::string(size_t)> kernelName;
          std::function<std::string(size_t)> moduleName;
          if (d->singleEntryPerModule()) {
            for (auto &[module_, data] : imageGroups)
              d->loadModule(module_, data);
            kernelName = [](auto) { return "main"; };
            moduleName = [](auto args) { return "arg" + std::to_string(args); };
          } else {
            // otherwise, we expect exactly one image
            if (imageGroups.size() != 1) {
              FAIL("Found more than one (" << imageGroups.size() << ") kernel test images for device `" << d->name() << "`(backend="
                                           << magic_enum::enum_name(backend) << ", features=" << (d->features() | mk_string(",")) << ")");
            } else {
              d->loadModule("module", imageGroups[0].second);
              kernelName = [](auto args) { return "_arg" + std::to_string(args); };
              moduleName = [](auto) { return "module"; };
            }
          }

          auto args = GENERATE(range(0, 28));
          DYNAMIC_SECTION("args=" << args) {
            auto q = d->createQueue(std::chrono::seconds(10));
            auto out_d = d->template mallocDeviceTyped<float>(1, Access::RW);

            float out = {};
            waitAll([&](auto &h) { q->enqueueHostToDeviceAsyncTyped(&out, out_d, 1, h); });

            size_t scalarArgCount = args == 0 ? 0 : args - 1;
            std::vector<float> values(scalarArgCount);
            std::iota(values.begin(), values.end(), 1);

            ArgBuffer buffer;

            if (d->sharedAddressSpace()) buffer.append(Type::IntS64, nullptr);

            for (size_t i = 0; i < scalarArgCount; ++i)
              buffer.append(Type::Float32, &values[i]);

            if (args != 0) buffer.append(Type::Ptr, &out_d);
            buffer.append(Type::Void, {});

            float expected = args == 0 ? 0                         // Actual 0 arg kernel, expect 0
                                       : (scalarArgCount == 0 ? 42 // 1 arg with no scalar, use constant
                                                              : std::reduce(values.begin(), values.end()));

            waitAll([&](auto &h) { q->enqueueInvokeAsync(moduleName(args), kernelName(args), buffer, {}, h); });
            waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1, h); });
            CHECK_THAT(out, Catch::Matchers::WithinULP(expected, 0));
            d->freeDevice(out_d);
          }
        } else {
          WARN("No kernel test image found for device `" << d->name() << "`(backend=" << magic_enum::enum_name(backend)
                                                         << ", features=" << (d->features() | mk_string(",")) << ")");
        }
      }
    }
  }
}

TEST_CASE("GPU Args") {
#ifndef NDEBUG
  WARN("Make sure ASAN is disabled otherwise most GPU backends will fail with memory related errors");
#endif
  ImageGroups images{};
  images.insert(generated::gpu::args.begin(), generated::gpu::args.end());
#ifdef RUNTIME_ENABLE_METAL
  images.insert(generated::msl::args.begin(), generated::msl::args.end());
#endif
  testArgs(images, //
           {
#ifndef __APPLE__
               Backend::CUDA,
               Backend::HIP,
               Backend::HSA,
#endif
               Backend::OpenCL,
#ifdef RUNTIME_ENABLE_METAL
               Backend::Metal,
#endif
           });
}

TEST_CASE("SPIRV Args") {
#ifdef __APPLE__
  WARN("Vulkan not natively supported on macOS");
#else
  testArgs(generated::spirv::glsl_args, //
           {Backend::Vulkan}            //
  );
#endif
}

TEST_CASE("CPU Args") {
  testArgs(generated::cpu::args,                               //
           {Backend::RelocatableObject, Backend::SharedObject} //
  );
}
