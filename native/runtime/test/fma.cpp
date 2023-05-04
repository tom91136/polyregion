#include "concurrency_utils.hpp"
#include "kernels/cpu_fma.hpp"
#include "kernels/gpu_fma.hpp"
#include "test_utils.h"
#include "utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace polyregion::runtime;
using namespace polyregion::test_utils;
using namespace polyregion::concurrency_utils;

const static std::vector<float> xs{0.f,
                                   -0.f,
                                   1.f,
                                   42.f,
                                   -42.f,                                 //
                                   std::numeric_limits<float>::epsilon(), //
                                   std::numeric_limits<float>::max(),     //
                                   std::numeric_limits<float>::min()};

TEST_CASE("CPU/GPU scalar fma") {
  auto backend = GENERATE(                                        //
      Backend::CUDA, Backend::OpenCL, Backend::HIP, Backend::HSA, //
      Backend::RELOCATABLE_OBJ, Backend::SHARED_OBJ               //
  );

  DYNAMIC_SECTION("backend=" << nameOfBackend(backend)) {
    auto platform = Platform::of(backend);
    for (auto &d : platform->enumerate()) {
      DYNAMIC_SECTION("device=" << d->name()) {
        if (auto image = findTestImage((backend == Backend::RELOCATABLE_OBJ || backend == Backend::SHARED_OBJ)
                                           ? generated::cpu::fma
                                           : generated::gpu::fma,
                                       backend, d->features());
            image) {
          d->loadModule("module", *image);
          auto a = GENERATE(from_range(xs));
          auto b = GENERATE(from_range(xs));
          auto c = GENERATE(from_range(xs));
          DYNAMIC_SECTION("a=" << a << " b=" << b << " c=" << c) {
            auto q = d->createQueue();
            auto out_d = d->mallocTyped<float>(1, Access::RW);

            ArgBuffer buffer;
            if (d->sharedAddressSpace()) buffer.append(Type::Long64, nullptr);

            buffer.append(
                {{Type::Float32, &a}, {Type::Float32, &b}, {Type::Float32, &c}, {Type::Ptr, &out_d}, {Type::Void, {}}});

            waitAll([&](auto &h) {
              q->enqueueInvokeAsync( //
                  "module", "_fma",
                  buffer, //
                  {}, h);
            });
            float out = {};
            waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1, h); });
            REQUIRE_THAT(out, Catch::Matchers::WithinULP(a * b + c, 0));
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
