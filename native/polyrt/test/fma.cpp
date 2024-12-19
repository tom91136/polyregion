#include "kernels/generated_cpu_fma.hpp"
#include "kernels/generated_gpu_fma.hpp"
#include "kernels/generated_msl_fma.hpp"
#include "kernels/generated_spirv_glsl_fma.hpp"
#include "polyregion/concurrency_utils.hpp"
#include "polyregion/io.hpp"
#include "polyregion/utils.hpp"
#include "test_utils.h"
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

template <typename I> void testFma(I images, std::initializer_list<Backend> backends) {
  auto backend = GENERATE_REF(values(backends));
  auto platform = polyregion::test_utils::makePlatform(backend);
  DYNAMIC_SECTION("backend=" << to_string(backend)) {
    for (auto &d : platform->enumerate()) {
      DYNAMIC_SECTION("device=" << d->name()) {
        if (auto imageGroups = findTestImage(images, backend, d->features()); !imageGroups.empty()) {

          if (imageGroups.size() != 1) {
            FAIL("Found more than one (" << imageGroups.size() << ") kernel test images for device `" << d->name()
                                         << "`(backend=" << to_string(backend) << ", features="
                                         << polyregion::mk_string<std::string>(d->features(), [](auto x) {  return x; }, ",") << ")");
          }

          std::string module_;
          std::string function_;
          if (d->singleEntryPerModule()) {
            module_ = "fma";
            function_ = "main";
          } else {
            module_ = "module";
            function_ = "_fma";
          }
          d->loadModule(module_, imageGroups[0].second);

          auto a = GENERATE(from_range(xs));
          auto b = GENERATE(from_range(xs));
          auto c = GENERATE(from_range(xs));
          DYNAMIC_SECTION("a=" << a << " b=" << b << " c=" << c) {
            auto q = d->createQueue();
            auto out_d = d->template mallocDeviceTyped<float>(1, Access::RW);
            //
            ArgBuffer buffer;
            if (d->sharedAddressSpace()) buffer.append(Type::IntS64, nullptr);

            buffer.append({{Type::Float32, &a}, {Type::Float32, &b}, {Type::Float32, &c}, {Type::Ptr, &out_d}, {Type::Void, {}}});

            waitAll([&](auto &h) {
              q->enqueueInvokeAsync( //
                  module_, function_,
                  buffer, //
                  {}, h);
            });
            float out = {};
            waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1, h); });
            auto expected = a * b + c;
            INFO("fma actual=" << out << " expected=" << expected);

            if (c == 0 && //
                ((a == std::numeric_limits<float>::min() && b == std::numeric_limits<float>::epsilon()) ||
                 (b == std::numeric_limits<float>::min() && a == std::numeric_limits<float>::epsilon()))) {
              CHECK_THAT(out, Catch::Matchers::WithinRel(0.f) || Catch::Matchers::WithinRel(expected));
            } else {
              CHECK_THAT(out, Catch::Matchers::WithinRel(expected));
            }

            d->freeDevice(out_d);
          }
        } else {
          WARN("No kernel test image found for device `"
               << d->name() << "`(backend=" << to_string(backend)
               << ", features=" << polyregion::mk_string<std::string>(d->features(), [](auto x) {  return x; }, ",") << ")");
        }
      }
    }
  }
}

TEST_CASE("GPU FMA") {
#ifndef NDEBUG
  WARN("Make sure ASAN is disabled otherwise most GPU backends will fail with memory related errors");
#endif
  polyregion::test_utils::ImageGroups images{};
  images.insert(generated::gpu::fma.begin(), generated::gpu::fma.end());
#ifdef RUNTIME_ENABLE_METAL
  images.insert(generated::msl::fma.begin(), generated::msl::fma.end());
#endif
  testFma(images, //
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

TEST_CASE("SPIRV FMA") {
#ifdef __APPLE__
  WARN("Vulkan not natively supported on macOS");
#else
  testFma(generated::spirv::glsl_fma, //
          {Backend::Vulkan}           //
  );
#endif
}

TEST_CASE("CPU FMA") {
  testFma(generated::cpu::fma,                                //
          {Backend::RelocatableObject, Backend::SharedObject} //
  );
}
