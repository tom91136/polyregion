#include "concurrency_utils.hpp"
#include "io.hpp"
#include "kernels/generated_cpu_args.hpp"
#include "kernels/generated_gpu_args.hpp"
#include "kernels/generated_spirv_glsl_args.hpp"
#include "test_utils.h"
#include "utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace polyregion::runtime;
using namespace polyregion::test_utils;
using namespace polyregion::concurrency_utils;

// See https://github.com/JuliaGPU/AMDGPU.jl/issues/10

template <typename I> void testArgs(I images, std::initializer_list<Backend> backends) {
  auto backend = GENERATE_REF(values(backends));
  auto platform = Platform::of(backend);
  DYNAMIC_SECTION("backend=" << nameOfBackend(backend)) {
    for (auto &d : platform->enumerate()) {
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
              FAIL("Found more than one ("
                   << imageGroups.size() << ") kernel test images for device `" << d->name()
                   << "`(backend=" << nameOfBackend(backend)
                   << ", features=" << polyregion::mk_string<std::string>(d->features(), std::identity(), ",") << ")");
            } else {
              d->loadModule("module", imageGroups[0].second);
              kernelName = [](auto args) { return "_arg" + std::to_string(args); };
              moduleName = [](auto) { return "module"; };
            }
          }

          auto args = GENERATE(range(0, 28));
          DYNAMIC_SECTION("args=" << args) {
            auto q = d->createQueue();
            auto out_d = d->template mallocTyped<float>(1, Access::RW);

            float out = {};
            waitAll([&](auto &h) { q->enqueueHostToDeviceAsyncTyped(&out, out_d, 1, h); });

            size_t scalarArgCount = args == 0 ? 0 : args - 1;
            std::vector<float> values(scalarArgCount);
            std::iota(values.begin(), values.end(), 1);

            ArgBuffer buffer;

            if (d->sharedAddressSpace()) buffer.append(Type::Long64, nullptr);

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
            d->free(out_d);
          }
        } else {
          WARN("No kernel test image found for device `"
               << d->name() << "`(backend=" << nameOfBackend(backend)
               << ", features=" << polyregion::mk_string<std::string>(d->features(), std::identity(), ",") << ")");
        }
      }
    }
  }
}

TEST_CASE("GPU Args") {
#ifndef NDEBUG
  WARN("Make sure ASAN is disabled otherwise most GPU backends will fail with memory related errors");
#endif
  testArgs(generated::gpu::args,                                        //
           {Backend::CUDA, Backend::OpenCL, Backend::HIP, Backend::HSA} //
  );
}

TEST_CASE("SPIRV Args") {
  testArgs(generated::spirv::glsl_args, //
           {Backend::Vulkan}            //
  );
}

TEST_CASE("Metal Args") {
  auto xs = polyregion::read_struct<uint8_t>("/Users/tom/polyregion/native/runtime/test/kernels/args.msl");
  const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>> stream_float = {
      {"Metal", {{"", xs}}}};


  testArgs(stream_float, //
          {Backend::Metal}           //
  );
}

TEST_CASE("CPU Args") {
  testArgs(generated::cpu::args,                           //
           {Backend::RELOCATABLE_OBJ, Backend::SHARED_OBJ} //
  );
}
