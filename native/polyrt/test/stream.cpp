#include "stream.hpp"
#include "io.hpp"
#include "kernels/generated_cpu_stream.hpp"
#include "kernels/generated_gpu_stream_double.hpp"
#include "kernels/generated_gpu_stream_float.hpp"
#include "kernels/generated_msl_stream_float.hpp"
#include "kernels/generated_spirv_glsl_stream.hpp"
#include "test_utils.h"
#include "utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace polyregion::runtime;

template <typename T, typename I>
void testStream(I images, Type tpe, const std::string &suffix, T relTolerance, //
                std::initializer_list<size_t> sizes,                           //
                std::initializer_list<size_t> groupSizes,                      //
                std::initializer_list<size_t> times,                           //
                std::initializer_list<Backend> backends) {
  auto backend = GENERATE_REF(values(backends));
  auto platform = Platform::of(backend);
  DYNAMIC_SECTION("backend=" << platform->name()) {
    auto size = GENERATE_REF(values(sizes));
    DYNAMIC_SECTION("size=" << size) {
      auto time = GENERATE_REF(values(times));
      DYNAMIC_SECTION("times=" << time) {
        auto Ncore = GENERATE_REF(values(groupSizes));
        DYNAMIC_SECTION("Ncore=" << Ncore) {
          for (auto &d : platform->enumerate()) {
            DYNAMIC_SECTION("device=" << d->name()) {
              if (auto imageGroups = polyregion::test_utils::findTestImage(images, backend, d->features());
                  !imageGroups.empty()) {

                polyregion::stream::Kernels<std::pair<std::string, std::string>> kernelSpecs;
                if (d->singleEntryPerModule()) {
                  for (auto &[module_, data] : imageGroups)
                    d->loadModule(module_, data);
                  kernelSpecs = {.copy = {"stream_copy" + suffix, "main"},
                                 .mul = {"stream_mul" + suffix, "main"},
                                 .add = {"stream_add" + suffix, "main"},
                                 .triad = {"stream_triad" + suffix, "main"},
                                 .dot = {"stream_dot" + suffix, "main"}};
                } else {
                  // otherwise, we expect exactly one image
                  if (imageGroups.size() != 1) {
                    FAIL("Found more than one ("
                         << imageGroups.size() << ") kernel test images for device `" << d->name()
                         << "`(backend=" << to_string(backend) << ", features="
                         << polyregion::mk_string<std::string>(d->features(), std::identity(), ",") << ")");
                  } else {
                    d->loadModule("module", imageGroups[0].second);
                    kernelSpecs = {.copy = {"module", "stream_copy" + suffix},
                                   .mul = {"module", "stream_mul" + suffix},
                                   .add = {"module", "stream_add" + suffix},
                                   .triad = {"module", "stream_triad" + suffix},
                                   .dot = {"module", "stream_dot" + suffix}};
                  }
                }

                polyregion::stream::runStream<T>(
                    tpe, size, time, Ncore, platform->name(), std::move(d), kernelSpecs, false,
                    [](auto actual, auto limit) {
                      INFO("array validation");
                      CHECK(actual < limit);
                    },
                    [&](auto actual) {
                      INFO("dot validation");
                      CHECK(actual < relTolerance);
                    });
              } else {
                WARN("No kernel test image found for device `"
                     << d->name() << "`(backend=" << to_string(backend) << ", features="
                     << polyregion::mk_string<std::string>(d->features(), std::identity(), ",") << ")");
              }
            }
          }
        }
      }
    }
  }
};

TEST_CASE("GPU BabelStream") {
#ifndef NDEBUG
  WARN("Make sure ASAN is disabled otherwise most GPU backends will fail with memory related errors");
#endif
  DYNAMIC_SECTION("float") {
    polyregion::test_utils::ImageGroups images{};
    images.insert(generated::gpu::stream_float.begin(), generated::gpu::stream_float.end());
#ifdef RUNTIME_ENABLE_METAL
    images.insert(generated::msl::stream_float.begin(), generated::msl::stream_float.end());
#endif
    testStream<float>(images, Type::Float32, "_float", 0.008f,               //
                      {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}, //
                      {1, 2, 32, 64, 128, 256},                              //
                      {1, 2, 10},                                            //
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
#ifndef __APPLE__ // macOS can't do doubles: Not supported in Metal and CL gives CL_INVALID_KERNEL
  DYNAMIC_SECTION("double") {
    testStream<double>(generated::gpu::stream_double, Type::Double64, "_double", 0.008f, //
                       {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},            //
                       {1, 2, 32, 64, 128, 256},                                         //
                       {1, 2, 10},                                                       //
                       {Backend::CUDA, Backend::OpenCL, Backend::HIP, Backend::HSA});
  }
#endif
}

TEST_CASE("SPIRV BabelStream") {
#ifdef __APPLE__
  WARN("Vulkan not natively supported on macOS");
#else

//  testStream<float>(generated::spirv::glsl_stream, Type::Float32, "_float", 0.008f, //
//                    {33554432},          //
//                    {256},                                       //
//                    {100},                                                     //
//                    {Backend::Vulkan});

  DYNAMIC_SECTION("float") {
    testStream<float>(generated::spirv::glsl_stream, Type::Float32, "_float", 0.008f, //
                      {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},          //
                      {1, 2, 32, 64, 128, 256},                                       //
                      {1, 2, 10},                                                     //
                      {Backend::Vulkan});
  }
  DYNAMIC_SECTION("double") {
    testStream<double>(generated::spirv::glsl_stream, Type::Double64, "_double", 0.008f, //
                       {1024, 2048},                                                     //
                       {1, 2, 32, 64},                                                   //
                       {1, 2, 10},                                                       //
                       {Backend::Vulkan});
  }
#endif
}

TEST_CASE("CPU BabelStream") {
  DYNAMIC_SECTION("double") {
    testStream<double>(generated::cpu::stream, Type::Double64, "_double", 0.0008, //
                       {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},     //
                       {1, 2, 3, 4, 5, 6, 7, 8},                                  //
                       {1, 2, 10},                                                //
                       {Backend::RelocatableObject, Backend::SharedObject}            //
    );
  }
  DYNAMIC_SECTION("float") {
    testStream<float>(generated::cpu::stream, Type::Float32, "_float", 0.008f, //
                      {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},   //
                      {1, 2, 3, 4, 5, 6, 7, 8},                                //
                      {1, 2, 10},                                              //
                      {Backend::RelocatableObject, Backend::SharedObject}          //
    );
  }
}
