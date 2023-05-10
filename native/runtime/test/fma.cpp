#include "concurrency_utils.hpp"
#include "io.hpp"
#include "kernels/generated_cpu_fma.hpp"
#include "kernels/generated_gpu_fma.hpp"
#include "kernels/generated_spirv_glsl_fma.hpp"
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

template <typename I> void testFma(I images, std::initializer_list<Backend> backends) {
  auto backend = GENERATE_REF(values(backends));
  auto platform = Platform::of(backend);
  DYNAMIC_SECTION("backend=" << nameOfBackend(backend)) {
    for (auto &d : platform->enumerate()) {
      DYNAMIC_SECTION("device=" << d->name()) {
        if (auto imageGroups = findTestImage(images, backend, d->features()); !imageGroups.empty()) {

          if (imageGroups.size() != 1) {
            FAIL("Found more than one (" << imageGroups.size() << ") kernel test images for device `" << d->name()
                                         << "`(backend=" << nameOfBackend(backend) << ", features="
                                         << polyregion::mk_string<std::string>(d->features(), std::identity(), ",")
                                         << ")");
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
            auto out_d = d->template mallocTyped<float>(1, Access::RW);
//
            ArgBuffer buffer;
            if (d->sharedAddressSpace()) buffer.append(Type::Long64, nullptr);

            buffer.append(
                {{Type::Float32, &a}, {Type::Float32, &b}, {Type::Float32, &c}, {Type::Ptr, &out_d}, {Type::Void, {}}});

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

TEST_CASE("GPU FMA") {
#ifndef NDEBUG
  WARN("Make sure ASAN is disabled otherwise most GPU backends will fail with memory related errors");
#endif
  testFma(generated::gpu::fma,                                         //
          {Backend::CUDA, Backend::OpenCL, Backend::HIP, Backend::HSA} //
  );
}

TEST_CASE("Metal FMA") {
  auto xs = polyregion::read_struct<uint8_t>("/Users/tom/polyregion/native/runtime/test/kernels/fma.msl");
  const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>> stream_float = {
      {"Metal", {{"", xs}}}};


  testFma(stream_float, //
          {Backend::Metal}           //
  );
}

TEST_CASE("SPIRV FMA") {
  testFma(generated::spirv::glsl_fma, //
          {Backend::Vulkan}           //
  );
}

TEST_CASE("CPU FMA") {
  testFma(generated::cpu::fma,                            //
          {Backend::RELOCATABLE_OBJ, Backend::SHARED_OBJ} //
  );
}
