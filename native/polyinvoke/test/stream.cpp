#include "polyregion/stream.hpp"

#include <cmath>

#include "aspartame/all.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_range.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/io.hpp"

#include "kernels/generated_cpu_stream.hpp"
#include "kernels/generated_gpu_stream_double.hpp"
#include "kernels/generated_gpu_stream_float.hpp"
#include "kernels/generated_msl_stream_float.hpp"
#include "kernels/generated_spirv_glsl_stream.hpp"
#include "kernels/generated_ze_stream_double.hpp"
#include "kernels/generated_ze_stream_float.hpp"
#include "test_utils.h"

using namespace polyregion::invoke;
using namespace aspartame;

template <typename T, typename I>
void testStream(I images, Type tpe, const std::string &suffix, T relTolerance, //
                std::initializer_list<size_t> sizes,                           //
                std::initializer_list<size_t> groupSizes,                      //
                std::initializer_list<size_t> times,                           //
                std::initializer_list<Backend> backends) {
  std::vector<Backend> enabled;
  for (auto b : backends)
    if (!polyregion::test_utils::isBackendDisabled(b)) enabled.push_back(b);
  if (enabled.empty()) return;
  auto backend = GENERATE_COPY(from_range(enabled));
  auto platform = polyregion::test_utils::makePlatform(backend);
  if (!platform) {
    WARN("Backend " << static_cast<int>(backend) << " is unavailable on this host - skipping");
    return;
  }

  DYNAMIC_SECTION("backend=" << platform->name()) {
    auto size = GENERATE_REF(values(sizes));
    DYNAMIC_SECTION("size=" << size) {
      auto time = GENERATE_REF(values(times));
      DYNAMIC_SECTION("times=" << time) {
        auto Ncore = GENERATE_REF(values(groupSizes));
        DYNAMIC_SECTION("Ncore=" << Ncore) {
          for (auto &d : platform->enumerate()) {
            if (polyregion::test_utils::isDeviceDisabled(d->name())) continue;
            const auto deviceFeatures = d->features();
            // XXX skip OpenCL SPIR-V-format duplicate; polyinvoke ships only source kernels.
            if (backend == Backend::OpenCL &&
                std::find(deviceFeatures.begin(), deviceFeatures.end(), "spirv_kernel") != deviceFeatures.end())
              continue;
            // XXX skip fp64 iterations on devices that don't expose fp64 (Vulkan/OpenCL/LevelZero
            // drivers tend to SEGV rather than reject cleanly).
            if (tpe == Type::Float64 && std::find(deviceFeatures.begin(), deviceFeatures.end(), "fp64") == deviceFeatures.end() &&
                (backend == Backend::Vulkan || backend == Backend::OpenCL || backend == Backend::LevelZero))
              continue;
            DYNAMIC_SECTION("device=" << d->name()) {
              auto _deviceLock = polyregion::test_utils::lockDevice(backend, *d);
              if (auto imageGroups = polyregion::test_utils::findTestImage(images, backend, deviceFeatures); !imageGroups.empty()) {

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
                    FAIL("Found more than one (" << imageGroups.size() << ") kernel test images for device `" << d->name()
                                                 << "`(backend=" << magic_enum::enum_name(backend)
                                                 << ", features=" << (d->features() | mk_string(",")) << ")");
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
                    tpe, size, time, Ncore, platform->name(), platform->kind(), *d, kernelSpecs, false,
                    [](auto actual, auto limit) {
                      INFO("array validation");
                      CHECK(actual < limit);
                    },
                    [&](auto actual) {
                      INFO("dot validation");
                      CHECK(actual < relTolerance);
                    });
              } else {
                WARN("No kernel test image found for device `" << d->name() << "`(backend=" << magic_enum::enum_name(backend)
                                                               << ", features=" << (d->features() | mk_string(",")) << ")");
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
    // Group sizes start at the warp size (32). Sub-warp groups (1, 2) are pathological for
    // GPU dispatch -- CUDA/HIP/Vulkan implementations don't optimise for them and some drivers
    // trigger illegal-address faults during the dot kernel's shared-memory reduction when the
    // block size is below the warp width. Threaded-CPU stream covers the small group sizes.
    testStream<float>(images, Type::Float32, "_float", 0.008f,               //
                      {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}, //
                      {32, 64, 128, 256},                                    //
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
    testStream<double>(generated::gpu::stream_double, Type::Float64, "_double", 0.008f, //
                       {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},           //
                       {32, 64, 128, 256},                                              //
                       {1, 2, 10},                                                      //
                       {Backend::OpenCL, Backend::CUDA, Backend::HIP, Backend::HSA});
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
    testStream<double>(generated::spirv::glsl_stream, Type::Float64, "_double", 0.008f, //
                       {1024, 2048},                                                    //
                       {1, 2, 32, 64},                                                  //
                       {1, 2, 10},                                                      //
                       {Backend::Vulkan});
  }
#endif
}

TEST_CASE("CPU BabelStream") {
  DYNAMIC_SECTION("double") {
    testStream<double>(generated::cpu::stream, Type::Float64, "_double", 0.0008, //
                       {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},    //
                       {1, 2, 3, 4, 5, 6, 7, 8},                                 //
                       {1, 2, 10},                                               //
                       {Backend::RelocatableObject, Backend::SharedObject}       //
    );
  }
  DYNAMIC_SECTION("float") {
    testStream<float>(generated::cpu::stream, Type::Float32, "_float", 0.008f, //
                      {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},   //
                      {1, 2, 3, 4, 5, 6, 7, 8},                                //
                      {1, 2, 10},                                              //
                      {Backend::RelocatableObject, Backend::SharedObject}      //
    );
  }
}

TEST_CASE("ZE BabelStream") {
  DYNAMIC_SECTION("float") {
    testStream<float>(generated::ze::stream_float, Type::Float32, "_float", 0.008f, //
                      {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},        //
                      {32, 64, 128, 256},                                           //
                      {1, 2, 10},                                                   //
                      {Backend::LevelZero});
  }
  DYNAMIC_SECTION("double") {
    testStream<double>(generated::ze::stream_double, Type::Float64, "_double", 0.008f, //
                       {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},          //
                       {32, 64, 128, 256},                                             //
                       {1, 2, 10},                                                     //
                       {Backend::LevelZero});
  }
}
