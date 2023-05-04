#include "concurrency_utils.hpp"
#include "kernels/cpu_args.hpp"
#include "kernels/gpu_args.hpp"
#include "test_utils.h"
#include "utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace polyregion::runtime;
using namespace polyregion::test_utils;
using namespace polyregion::concurrency_utils;

// See https://github.com/JuliaGPU/AMDGPU.jl/issues/10

TEST_CASE("CPU/GPU scalar args") {
  auto backend = GENERATE(                                        //
      Backend::CUDA, Backend::OpenCL, Backend::HIP, Backend::HSA, //
      Backend::RELOCATABLE_OBJ, Backend::SHARED_OBJ               //
  );

  DYNAMIC_SECTION("backend=" << nameOfBackend(backend)) {
    auto platform = Platform::of(backend);

    for (auto &d : platform->enumerate()) {
      DYNAMIC_SECTION("device=" << d->name()) {
        if (auto image = findTestImage((backend == Backend::RELOCATABLE_OBJ || backend == Backend::SHARED_OBJ)
                                           ? generated::cpu::args
                                           : generated::gpu::args,
                                       backend, d->features());
            image) {
          d->loadModule("module", *image);
          auto args = GENERATE(range(0, 28));
          DYNAMIC_SECTION("args=" << args) {
            auto q = d->createQueue();
            auto out_d = d->mallocTyped<float>(1, Access::RW);

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

            waitAll([&](auto &h) { q->enqueueInvokeAsync("module", "_arg" + std::to_string(args), buffer, {}, h); });
            waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1, h); });
            REQUIRE_THAT(out, Catch::Matchers::WithinULP(expected, 0));
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